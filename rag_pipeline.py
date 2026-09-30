import os
import re
import chromadb
from chromadb.utils import embedding_functions
import hashlib
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Tuple, List

logger = logging.getLogger(__name__)

# Explicit parameter decisions:
# CHUNK_SIZE = 350 words (~450 tokens): EU-MDR legal clauses average 200–300 words. 
#   A 350-word window captures full statutory obligations (e.g., Art 10(9) QMS requirements) 
#   without mixing unrelated articles or diluting dense legal terms.
# - OVERLAP = 50 words (~15%): Preserves conditional sub-clauses (e.g., exceptions in 
#   paragraph 2) if a split lands across paragraph boundaries.
DEFAULT_CHUNK_SIZE = 350
DEFAULT_OVERLAP = 50
DB_PATH = os.environ.get("CHROMA_PERSIST_DIRECTORY") or os.environ.get("CHROMA_DB_DIR") or "./chroma_db"
COLLECTION_NAME = "document_collection"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"  # Sentence-Transformers model for semantic embeddings

client = None
collection = None

def compute_file_sha256(filepath: Path) -> str:
    """Computes exact SHA-256 byte digest for content drift detection."""
    sha256 = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            sha256.update(chunk)
    return sha256.hexdigest()

def get_registry_path(persist_dir: str = None) -> Path:
    base_dir = Path(persist_dir or DB_PATH)
    return base_dir / "registry.json"

def write_feature_registry(
    persist_dir: str,
    source_file: Path,
    chunk_count: int,
    embedding_dimension: int,
) -> Path:
    registry_data: Dict[str, Any] = {
        "embedding_model": EMBEDDING_MODEL_NAME,
        "embedding_dimension": embedding_dimension,
        "chunk_size": DEFAULT_CHUNK_SIZE,
        "chunk_overlap": DEFAULT_OVERLAP,
        "source_file": source_file.name,
        "source_content_hash": compute_file_sha256(source_file),
        "chunk_count": chunk_count,
        "ingested_at": datetime.now(timezone.utc).isoformat(),
    }
    
    target_path = get_registry_path(persist_dir)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    with open(target_path, "w", encoding="utf-8") as f:
        json.dump(registry_data, f, indent=2)
    
    logger.info("Feature registry written to %s", target_path)
    return target_path

# 1. Initialize embedding function (Sentence-Transformers)
def init_store(persist_directory: str = None):
    """Initializes the embedding model and Chroma collection at server startup."""
    global client, collection
    
    target_path = persist_directory or os.environ.get("CHROMA_PERSIST_DIRECTORY") or os.environ.get("CHROMA_DB_DIR") or DB_PATH

    # If collection is already initialized with the same target path, reuse it
    if collection is not None and client is not None:
        return collection

    embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL_NAME
    )
    client = chromadb.PersistentClient(path=target_path)
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_fn,
        metadata={"hnsw:space": "cosine"},
    )
    return collection

# 3. Chunking utility
def chunk_text(text: str, chunk_size: int = DEFAULT_CHUNK_SIZE, overlap: int = DEFAULT_OVERLAP) -> list[str]:
    words = text.split()
    if not words:
        return []
    
    chunks = []
    step = chunk_size - overlap
    for i in range(0, len(words), step):
        chunk = " ".join(words[i : i + chunk_size])
        chunks.append(chunk)
    return chunks

# 3b. Structural boundary detection (Article / Annex headers, not inline cross-references)
SECTION_HEADER_RE = re.compile(r'(?m)^(Article \d+[a-z]?|ANNEX [IVXLC]+)\s*\n([A-Z][^\n]*)')

def split_into_sections(text: str) -> list[tuple[str, str]]:
    matches = list(SECTION_HEADER_RE.finditer(text))
    if not matches:
        return [("Document", text)]

    sections = []
    if matches[0].start() > 0:
        sections.append(("Preamble", text[:matches[0].start()]))

    for i, m in enumerate(matches):
        label = f"{m.group(1)} - {m.group(2).strip()}"
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        sections.append((label, text[start:end]))

    return sections

# 4. Ingestion Workflow
def ingest_document(source_path: str, persist_directory: str = None) -> int:
    """
    Ingests source text preserving statutory Article/Annex section boundaries,
    indexes them into ChromaDB, and writes registry.json strictly on success.
    """
    col = init_store(persist_directory)
    file_path = Path(source_path)
    if not file_path.exists():
        raise FileNotFoundError(f"Source file {source_path} does not exist.")

    with open(file_path, "r", encoding="utf-8") as f:
        raw_text = f.read()

    # 1. Parse statutory boundaries using existing domain parsers
    sections = split_into_sections(raw_text)

    all_chunks: List[str] = []
    chunk_ids: List[str] = []
    metadatas: List[Dict[str, Any]] = []

    for section_title, section_body in sections:
        # Window within statutory section boundaries
        sub_chunks = chunk_text(
            section_body,
            chunk_size=DEFAULT_CHUNK_SIZE,
            overlap=DEFAULT_OVERLAP,
        )
        for chunk in sub_chunks:
            # Prepend statutory header to text for embedding/generation clarity
            formatted_text = f"[{section_title}]\n{chunk}"
            chunk_id = f"doc_chunk_{len(all_chunks)}"
            all_chunks.append(formatted_text)
            chunk_ids.append(chunk_id)
            metadatas.append(
                {
                    "source": file_path.name,
                    "section": section_title,
                    "chunk_index": len(all_chunks) - 1,
                }
            )

    if not all_chunks:
        logger.warning("No chunks generated from %s", source_path)
        return 0

    # 2. Derive embedding dimension via live forward pass on chunk 0
    probe_vector = col._embedding_function([all_chunks[0]])
    actual_dimension = len(probe_vector[0])

    # 3. Add to Chroma in batches
    batch_size = 100
    for i in range(0, len(all_chunks), batch_size):
        end_idx = min(i + batch_size, len(all_chunks))
        col.add(
            ids=chunk_ids[i:end_idx],
            documents=all_chunks[i:end_idx],
            metadatas=metadatas[i:end_idx],
        )

    # 4. ORDERING INVARIANT: Only record registry once insertion succeeds
    target_dir = persist_directory or DB_PATH
    write_feature_registry(
        persist_dir=target_dir,
        source_file=file_path,
        chunk_count=len(all_chunks),
        embedding_dimension=actual_dimension,
    )

    return len(all_chunks)

# 5. Query Function
def query_store(query_string: str, top_k: int = 3, min_similarity: float = 0.45) -> list[dict]:
    global collection
    if collection is None:
        init_store()

    results = collection.query(
        query_texts=[query_string],
        n_results=top_k,
        include=["documents", "distances", "metadatas"]
    )

    print(f"\n--- Top {top_k} Results for Query: '{query_string}' ---")
    output = []
    if not results["ids"] or not results["ids"][0]:
        return []
    
    for idx in range(len(results["ids"][0])):
        chunk_id = results["ids"][0][idx]
        distance = results["distances"][0][idx]
        similarity_score = 1 - distance  # Cosine similarity conversion
        doc = results["documents"][0][idx]
        meta = results["metadatas"][0][idx]

        if similarity_score < min_similarity:
            continue  # Skip low-similarity results

        print(f"\nRank {idx + 1} | Chunk ID: {chunk_id} | Similarity Score: {similarity_score:.4f}")
        print(f"Metadata: {meta}")
        print(f"Text Snippet:\n{doc[:200]}...")

        output.append({
            "chunk_id": chunk_id,
            "similarity_score": similarity_score,
            "section": meta.get("section", ""),
            "text": doc,
        })
    return output

def verify_feature_registry(
    persist_dir: str = None,
    source_file: Path = None,
) -> Tuple[bool, List[str]]:
    """
    Compares stored registry metadata against current runtime constants and source file hash.
    Resolves the source file from registry metadata if not explicitly provided.
    """
    base_dir = Path(persist_dir or DB_PATH)
    reg_path = get_registry_path(persist_dir)

    if not reg_path.exists():
        return False, [f"Registry file not found at {reg_path}"]

    try:
        with open(reg_path, "r", encoding="utf-8") as f:
            stored = json.load(f)
    except Exception as err:
        return False, [f"Failed to parse registry at {reg_path}: {err}"]

    mismatches: List[str] = []

    # 1. Chunk size check
    if stored.get("chunk_size") != DEFAULT_CHUNK_SIZE:
        mismatches.append(
            f"chunk_size mismatch: registry={stored.get('chunk_size')} vs current_runtime={DEFAULT_CHUNK_SIZE}"
        )

    # 2. Chunk overlap check
    if stored.get("chunk_overlap") != DEFAULT_OVERLAP:
        mismatches.append(
            f"chunk_overlap mismatch: registry={stored.get('chunk_overlap')} vs current_runtime={DEFAULT_OVERLAP}"
        )

    # 3. Embedding model check
    if stored.get("embedding_model") != EMBEDDING_MODEL_NAME:
        mismatches.append(
            f"embedding_model mismatch: registry={stored.get('embedding_model')} vs current_runtime={EMBEDDING_MODEL_NAME}"
        )

    # 4. Resolve source file from registry metadata if not overridden
    target_source = source_file
    if target_source is None:
        raw_source_name = stored.get("source_file")
        if not raw_source_name:
            mismatches.append("registry.json is missing required 'source_file' field.")
            return False, mismatches

        # Resolve candidate relative to CWD, base_dir, and base_dir's parent
        candidate = Path(raw_source_name)
        if not candidate.exists():
            candidate = base_dir / raw_source_name
        if not candidate.exists() and base_dir.parent.exists():
            candidate = base_dir.parent / raw_source_name

        target_source = candidate

    # 5. Content hash verification
    if target_source.exists():
        current_hash = compute_file_sha256(target_source)
        stored_hash = stored.get("source_content_hash")
        if current_hash != stored_hash:
            mismatches.append(
                f"source_content_hash mismatch for {target_source.name}: "
                f"registry={stored_hash} vs disk={current_hash}"
            )
    else:
        mismatches.append(f"Recorded source file '{target_source}' does not exist on disk.")

    return len(mismatches) == 0, mismatches

if __name__ == "__main__":
    DOC_PATH = "eu_mdr_text.txt"
    
    if os.path.exists(DOC_PATH):
        # -------------------------------------------------------------
        # PRE-INSTANTIATION CHECK: Evaluate sections before ChromaDB runs
        # -------------------------------------------------------------
        with open(DOC_PATH, "r", encoding="utf-8") as f:
            raw_text = f.read()

        pre_check_sections = split_into_sections(raw_text)
        distinct_sections = set(label for label, _ in pre_check_sections)

        print("\n================ PRE-INGESTION REPORT ================")
        print(f"Raw Document Character Count : {len(raw_text)}")
        print(f"Total Structural Sections    : {len(pre_check_sections)}")
        print(f"Distinct Section Headings   : {len(distinct_sections)}")
        print("======================================================\n")

        # Sanity Check Guardrail
        if len(distinct_sections) == 1 and list(distinct_sections)[0] == "Document":
            print("[WARNING] Regex matched 0 section headers! Every chunk will be tagged as 'Document'.")
            print("[WARNING] Check your SECTION_HEADER_RE pattern against 'eu_mdr_text.txt'.\n")
        else:
            print("[SUCCESS] Regex matched structural headers successfully. Proceeding to database setup.\n")

        # -------------------------------------------------------------
        # CHROMADB INSTANTIATION & INGESTION
        # -------------------------------------------------------------
        # Initialize store first so collection is not None
        init_store()

        # Only ingest if the collection is currently empty
        if collection.count() == 0:
            print("[INFO] Collection empty. Ingesting documents...")
            ingest_document(DOC_PATH)
        else:
            print(f"[INFO] Collection already contains {collection.count()} chunks.")

        # Test verification query
        test_query = "serious incident reporting vigilance timelines manufacturer obligations"
        query_store(test_query, top_k=3)
    else:
        print(f"[ERROR] Source file '{DOC_PATH}' not found.")
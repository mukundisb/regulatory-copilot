import os
import re
import chromadb
from chromadb.utils import embedding_functions

# Explicit parameter decisions:
# CHUNK_SIZE = 350 words (~450 tokens): EU-MDR legal clauses average 200–300 words. 
#   A 350-word window captures full statutory obligations (e.g., Art 10(9) QMS requirements) 
#   without mixing unrelated articles or diluting dense legal terms.
# - OVERLAP = 50 words (~15%): Preserves conditional sub-clauses (e.g., exceptions in 
#   paragraph 2) if a split lands across paragraph boundaries.
DEFAULT_CHUNK_SIZE = 350
DEFAULT_OVERLAP = 50

# 1. Initialize local embedding function (Sentence-Transformers)
embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)

# 2. Set up local persistent Chroma client
DB_PATH = "./chroma_db"
client = chromadb.PersistentClient(path=DB_PATH)

collection = client.get_or_create_collection(
    name="document_collection",
    embedding_function=embedding_fn,
    metadata={"hnsw:space": "cosine"}  # Using Cosine distance for standard normalized scoring
)

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
def ingest_document(file_path: str):
    with open(file_path, "r", encoding="utf-8") as f:
        raw_text = f.read()

    sections = split_into_sections(raw_text)

    ids, documents, metadatas = [], [], []
    for label, body in sections:
        for chunk in chunk_text(body, chunk_size=DEFAULT_CHUNK_SIZE, overlap=DEFAULT_OVERLAP):
            chunk_id = len(ids)
            ids.append(f"doc_chunk_{chunk_id}")
            documents.append(f"[{label}]\n{chunk}")
            metadatas.append({
                "chunk_index": chunk_id,
                "section": label,
                "total_words": len(chunk.split()),
            })

    collection.add(
        documents=documents,
        ids=ids,
        metadatas=metadatas
    )
    print(f"Ingested {len(documents)} chunks across {len(sections)} sections into ChromaDB at '{DB_PATH}'.")

# 5. Query Function
def query_store(query_string: str, top_k: int = 3):
    results = collection.query(
        query_texts=[query_string],
        n_results=top_k,
        include=["documents", "distances", "metadatas"]
    )

    print(f"\n--- Top {top_k} Results for Query: '{query_string}' ---")
    for idx in range(len(results["ids"][0])):
        chunk_id = results["ids"][0][idx]
        distance = results["distances"][0][idx]
        similarity_score = 1 - distance  # Cosine similarity conversion
        doc = results["documents"][0][idx]
        meta = results["metadatas"][0][idx]

        print(f"\nRank {idx + 1} | Chunk ID: {chunk_id} | Similarity Score: {similarity_score:.4f}")
        print(f"Metadata: {meta}")
        print(f"Text Snippet:\n{doc[:200]}...")

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
        # Only ingest if the collection is currently empty
        if collection.count() == 0:
            ingest_document(DOC_PATH)

    else:
        print(f"[ERROR] Source file '{DOC_PATH}' not found.")

    # Execution Query
    test_query = "In what language must device labels, packaging information, and instructions for use be provided?"
    query_store(test_query, top_k=3)
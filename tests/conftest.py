import os
import shutil
import pytest
import joblib
from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

import rag_pipeline

MODEL_DIR = "maude_classifier/model"
MODEL_FILE = os.path.join(MODEL_DIR, "maude_classifier.joblib")


@pytest.fixture(scope="session", autouse=True)
def setup_test_classifier_model():
    """Builds and serializes a calibrated classifier for all decision branches."""
    created = False
    if not os.path.exists(MODEL_FILE):
        os.makedirs(MODEL_DIR, exist_ok=True)
        
        pipe = Pipeline([
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2))),
            ("clf", LogisticRegression(C=10.0))
        ])
        
        # Explicit anchors covering all test payloads
        X_train = [
            # Death (D)
            "Patient died of acute myocardial infarction following catheter fracture and embolization",
            "Patient passed away during surgery after stent dislodged death fatal outcome cardiac arrest",
            "catheter fracture embolization fatal mortality death expired",
            # Injury (I)
            "Patient experienced severe allergic reaction following administration hospitalization",
            "blood loss hemorrhage emergency resuscitation severe patient injury physical trauma",
            # Malfunction (M)
            "Infusion pump screen froze and stopped delivery of medication, showing error code E-402",
            "During procedure, ventilator display flashed error code E-102 and stopped oxygen delivery",
            "Display went blank during self-test routine device malfunction component failure",
            "device malfunction error code stopped delivery screen froze mechanical failure",
            # Other (O)
            "routine inquiry packaging issue question regarding label user manual",
            "general question maintenance query other non-adverse event"
        ]
        y_train = ["D", "D", "D", "I", "I", "M", "M", "M", "M", "O", "O"]
        
        pipe.fit(X_train, y_train)
        joblib.dump(pipe, MODEL_FILE)
        created = True

    yield

    if created and os.path.exists(MODEL_FILE):
        try:
            os.remove(MODEL_FILE)
        except OSError:
            pass


@pytest.fixture(scope="session", autouse=True)
def setup_session_golden_chroma_db(tmp_path_factory):
    """
    Session-scoped: Ingests the canonical eu_mdr_text.txt once into a golden directory.
    All tests clone this golden directory, giving every test the full 412 chunks 
    and a valid registry.json without re-running ingestion repeatedly.
    """
    golden_dir = tmp_path_factory.mktemp("golden_chroma_db")
    corpus_path = Path("eu_mdr_text.txt")
    
    rag_pipeline.client = None
    rag_pipeline.collection = None

    if corpus_path.exists():
        # Ingest the real corpus once for the entire session (~8s)
        rag_pipeline.ingest_document(str(corpus_path), persist_directory=str(golden_dir))
    else:
        # Fallback to synthetic if eu_mdr_text.txt is missing
        collection = rag_pipeline.init_store(persist_directory=str(golden_dir))
        docs = [
            "Article 10 - General obligations of manufacturers regarding quality management systems.",
            "Article 87 - Reporting of serious incidents and field safety corrective actions vigilance timelines.",
            "Article 88 - Trend reporting for medical devices.",
            "Article 89 - Analysis of serious incidents and corrective actions by competent authorities."
        ]

        # Write actual fixture bytes to disk so compute_file_sha256 does not raise FileNotFoundError
        fallback_corpus_file = golden_dir / "test.txt"
        fallback_corpus_file.write_text("\n\n".join(docs), encoding="utf-8")

        
        collection.add(
            ids=[f"doc_chunk_{i}" for i in range(len(docs))],
            documents=docs,
            metadatas=[{"source": "test.txt", "section": d.split(" - ")[0], "chunk_index": i} for i, d in enumerate(docs)]
        )
        probe = collection._embedding_function([docs[0]])
        rag_pipeline.write_feature_registry(
            persist_dir=str(golden_dir),
            source_file=golden_dir / "test.txt",
            chunk_count=len(docs),
            embedding_dimension=len(probe[0]),
        )

    yield golden_dir

    rag_pipeline.client = None
    rag_pipeline.collection = None


@pytest.fixture(autouse=True)
def isolate_test_chroma(tmp_path, setup_session_golden_chroma_db, monkeypatch):
    """
    Function-scoped: Clones the pre-seeded session golden store (with its registry.json)
    into an isolated per-test directory. Every test starts with count > 0,
    actively exercising verify_feature_registry() without re-embedding eu_mdr_text.txt.
    """
    test_db_dir = tmp_path / "test_chroma_db"
    shutil.copytree(setup_session_golden_chroma_db, test_db_dir, dirs_exist_ok=True)

    # Point environment and pipeline to this test's isolated directory
    monkeypatch.setenv("CHROMA_PERSIST_DIRECTORY", str(test_db_dir))
    monkeypatch.setenv("CHROMA_DB_DIR", str(test_db_dir))
    monkeypatch.setattr(rag_pipeline, "DB_PATH", str(test_db_dir))

    # Reset singletons so Chroma re-binds to the function's isolated store
    rag_pipeline.client = None
    rag_pipeline.collection = None

    yield str(test_db_dir)

    rag_pipeline.client = None
    rag_pipeline.collection = None
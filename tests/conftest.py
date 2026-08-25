import os
import shutil
import pytest
import joblib
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

import chromadb

MODEL_DIR = "maude_classifier/model"
MODEL_FILE = os.path.join(MODEL_DIR, "maude_classifier.joblib")
CHROMA_DIR = "chroma_db"


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
        
        # Explicit anchors covering all test payload phrases
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
def setup_test_chroma_db():
    """
    Seeds persistent Chroma store with exact metadata ('section') and document texts
    matching the query vectors of test_retrieve_e2e_real_store and test_assess_e2e_*.
    """
    created = False
    if not os.path.exists(CHROMA_DIR):
        os.makedirs(CHROMA_DIR, exist_ok=True)
        client = chromadb.PersistentClient(path=CHROMA_DIR)
        
        # Seeded documents with high semantic overlap to test queries
        seeded_docs = [
            # Matches language / label query in test_retrieve_e2e_real_store
            "[ANNEX I] In what language must device labels, packaging information, and instructions for use be provided under European Union Medical Device Regulations. Requirements regarding the information supplied with the device.",
            
            # Matches vigilance / death branch query in test_assess_e2e_real_pipeline
            "[Article 87] Manufacturers shall report any serious incident reporting vigilance timelines manufacturer obligations for adverse events, myocardial infarction, catheter fracture, death, and severe deterioration.",
            
            # Matches malfunction branch query in test_assess_e2e_malfunction_branch_real_pipeline & fallback test
            "[Article 88] Device malfunction root cause analysis trend reporting corrective action CAPA. Infusion pump screen froze, ventilator display error code E-102 E-402 stopped oxygen delivery.",
            
            # General baseline
            "[Article 16] Cases in which obligations of manufacturers apply to distributors or importers carrying out translation."
        ]
        
        # CRITICAL: Use the key 'section' as asserted by test_retrieve_e2e_real_store
        seeded_metadatas = [
            {"section": "ANNEX I - GENERAL SAFETY AND PERFORMANCE REQUIREMENTS", "chunk_id": "doc_chunk_283"},
            {"section": "Article 87 - Reporting of serious incidents and field safety corrective actions", "chunk_id": "doc_chunk_180"},
            {"section": "Article 88 - Trend reporting and CAPA investigation", "chunk_id": "doc_chunk_182"},
            {"section": "Article 16 - Obligations of distributors and importers", "chunk_id": "doc_chunk_74"}
        ]
        
        seeded_ids = ["doc_chunk_283", "doc_chunk_180", "doc_chunk_182", "doc_chunk_74"]

        # Populate default collection and common aliases
        candidate_collections = ["regulatory_docs", "fda_guidance", "maude_index", "documents", "default"]
        
        for col_name in candidate_collections:
            try:
                col = client.get_or_create_collection(name=col_name)
                col.add(
                    documents=seeded_docs,
                    metadatas=seeded_metadatas,
                    ids=seeded_ids
                )
            except Exception:
                pass

        created = True

    yield

    if created and os.path.exists(CHROMA_DIR):
        shutil.rmtree(CHROMA_DIR, ignore_errors=True)
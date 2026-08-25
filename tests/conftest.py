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
    """Builds and serializes a robust dummy classifier for all decision branches."""
    created = False
    if not os.path.exists(MODEL_FILE):
        os.makedirs(MODEL_DIR, exist_ok=True)
        
        pipe = Pipeline([
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2))),
            ("clf", LogisticRegression(C=10.0))
        ])
        
        # Explicit, heavily anchored training samples for D, I, M, O
        X_train = [
            # Death (D)
            "acute cardiac arrest patient died fatal outcome death mortality expired fatal rupture",
            "patient deceased during surgery cardiac arrest following catheter rupture",
            # Injury (I)
            "patient sustained serious injury severe physical trauma required medical intervention hospitalization",
            "blood loss hemorrhage emergency resuscitation severe injury deterioration",
            # Malfunction (M)
            "device malfunction mechanical failure balloon burst component crack software error unexpected shutdown",
            "catheter balloon rupture failed deployment device defect malfunction without patient injury",
            "pump stopped working sensor error calibration fault malfunction",
            # Other (O)
            "routine inquiry packaging issue question regarding label user manual",
            "general question periodic review maintenance query other"
        ]
        y_train = ["D", "D", "I", "I", "M", "M", "M", "O", "O"]
        
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
    Creates persistent Chroma collections seeded with documents matching all search paths.
    """
    created = False
    if not os.path.exists(CHROMA_DIR):
        os.makedirs(CHROMA_DIR, exist_ok=True)
        client = chromadb.PersistentClient(path=CHROMA_DIR)
        
        seeded_docs = [
            "Serious incident reporting vigilance timelines manufacturer obligations for adverse events and mortality.",
            "Device malfunction root cause analysis trend reporting corrective action and preventive vigilance CAPA.",
            "Catheter balloon rupture during angioplasty procedure guidance and vigilance reporting.",
            "General regulatory vigilance fallback procedures when specific criteria are ambiguous."
        ]
        
        seeded_metadatas = [
            {"source": "MDR_Art_87", "category": "vigilance", "topic": "death_injury"},
            {"source": "FDA_21CFR803", "category": "malfunction", "topic": "malfunction"},
            {"source": "Clinical_Guidance", "category": "procedure", "topic": "angioplasty"},
            {"source": "Fallback_Guidance", "category": "fallback", "topic": "fallback"}
        ]
        
        seeded_ids = ["doc_1", "doc_2", "doc_3", "doc_4"]

        # Seed across all common collection names used by regulatory copilot architectures
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
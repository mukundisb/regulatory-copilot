import os
import shutil
import pytest
import joblib
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

import chromadb
from chromadb.api.types import Documents, EmbeddingFunction, Embeddings

MODEL_DIR = "maude_classifier/model"
MODEL_FILE = os.path.join(MODEL_DIR, "maude_classifier.joblib")
CHROMA_DIR = "chroma_db"


# 1. Deterministic Local Embedding Function (Zero Hugging Face Downloads)
class DeterministicTestEmbeddingFunction(EmbeddingFunction):
    """
    Generates deterministic, normalized 384-dim embeddings based on 
    keyword hashing to avoid model weight downloads and ensure reproducible scores.
    """
    def __call__(self, input: Documents) -> Embeddings:
        embeddings = []
        for text in input:
            t = text.lower()
            vec = np.zeros(384, dtype=np.float32)
            # Encode semantic clusters onto specific orthogonal dimensions
            if "serious incident" in t or "vigilance" in t or "death" in t or "cardiac" in t:
                vec[0] = 1.0
            elif "malfunction" in t or "root cause" in t or "balloon rupture" in t:
                vec[1] = 1.0
            elif "unrelated" in t or "fallback" in t:
                vec[2] = 1.0
            else:
                vec[3] = 1.0
            
            # Unit-normalize vector for cosine/L2 distance consistency
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec = vec / norm
            embeddings.append(vec.tolist())
        return embeddings


# 2. Fixture: Dummy Classifier Model
@pytest.fixture(scope="session", autouse=True)
def setup_test_classifier_model():
    """Builds and serializes a lightweight model to satisfy startup checks."""
    created = False
    if not os.path.exists(MODEL_FILE):
        os.makedirs(MODEL_DIR, exist_ok=True)
        pipe = Pipeline([
            ("tfidf", TfidfVectorizer()),
            ("clf", LogisticRegression())
        ])
        
        # Train on texts matching the 4 decision classes
        X_train = [
            "acute cardiac arrest catheter balloon rupture death",
            "severe patient injury during procedure",
            "device malfunction alert component failure",
            "routine inquiry non-serious general event"
        ]
        y_train = ["D", "I", "M", "O"]
        pipe.fit(X_train, y_train)
        
        joblib.dump(pipe, MODEL_FILE)
        created = True

    yield

    if created and os.path.exists(MODEL_FILE):
        try:
            os.remove(MODEL_FILE)
        except OSError:
            pass


# 3. Fixture: Seeded Chroma DB Collection
@pytest.fixture(scope="session", autouse=True)
def setup_test_chroma_db():
    """
    Creates and seeds the local Chroma DB directory with documents matching
    the exact query steering and threshold assertions.
    """
    created = False
    if not os.path.exists(CHROMA_DIR):
        os.makedirs(CHROMA_DIR, exist_ok=True)
        client = chromadb.PersistentClient(path=CHROMA_DIR)
        
        # Initialize or fetch default collection
        embed_fn = DeterministicTestEmbeddingFunction()
        collection = client.get_or_create_collection(
            name="regulatory_docs",
            embedding_function=embed_fn,
            metadata={"hnsw:space": "cosine"}
        )
        
        # Seeded documents matching the parameterized test signatures
        seeded_docs = [
            # D / I Branch Document
            "serious incident reporting vigilance timelines manufacturer obligations for adverse death and injury events",
            # M Branch Document
            "device malfunction root cause analysis trend reporting corrective action and preventive vigilance",
            # Fallback Reference Document
            "general regulatory vigilance fallback procedures when specific criteria are ambiguous",
            # Unrelated Baseline Document
            "unrelated general device operating manual and standard packaging details"
        ]
        
        seeded_metadatas = [
            {"branch": "D_I", "regulation": "MDR Article 87", "confidence_level": "high"},
            {"branch": "M", "regulation": "FDA 21 CFR 803", "confidence_level": "high"},
            {"branch": "fallback", "regulation": "Standard Guidance", "confidence_level": "medium"},
            {"branch": "unrelated", "regulation": "None", "confidence_level": "low"}
        ]
        
        seeded_ids = ["doc_vigilance_di", "doc_malfunction_m", "doc_fallback_general", "doc_unrelated_01"]
        
        collection.add(
            documents=seeded_docs,
            metadatas=seeded_metadatas,
            ids=seeded_ids
        )
        created = True

    yield

    if created and os.path.exists(CHROMA_DIR):
        shutil.rmtree(CHROMA_DIR, ignore_errors=True)
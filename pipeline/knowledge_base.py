"""
knowledge_base.py
==================
Diagram steps:  Knowledge Base (WHO, CDC, ICMR, PubMed)  ->  Vector Index (FAISS / ChromaDB)

Job of this file:
1. Load the reference text files in data/knowledge_docs/ (this is our stand-in
   "knowledge base" — in production you'd point this at real WHO/CDC/ICMR/
   PubMed sources).
2. Turn them into a searchable "vector index" so the retriever can find the
   most relevant snippet for any given test.

We use scikit-learn's TF-IDF vectorizer + cosine similarity here because it
needs no extra downloads and is easy to read line-by-line. The diagram calls
for FAISS / ChromaDB with real embeddings — see `FaissVectorIndex` below for
a drop-in upgrade if you install `faiss-cpu` and `sentence-transformers`.
"""

import glob
import os
from dataclasses import dataclass
from typing import List

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import config


@dataclass
class KnowledgeDoc:
    source_file: str
    text: str


class VectorIndex:
    """
    A simple, dependency-light vector index.

    Loads every .txt file in the knowledge_docs folder, vectorizes them with
    TF-IDF, and can return the top-k most similar documents for a query.
    """

    def __init__(self, docs_dir: str = config.KNOWLEDGE_DOCS_DIR):
        self.docs: List[KnowledgeDoc] = _load_documents(docs_dir)
        self._vectorizer = TfidfVectorizer(stop_words="english")
        self._doc_matrix = self._vectorizer.fit_transform(
            [doc.text for doc in self.docs]
        )

    def search(self, query: str, top_k: int = config.TOP_K_RESULTS) -> List[KnowledgeDoc]:
        """Return the top_k documents most relevant to `query`."""
        query_vector = self._vectorizer.transform([query])
        similarities = cosine_similarity(query_vector, self._doc_matrix)[0]

        # argsort ascending, so take the last top_k and reverse for best-first
        ranked_indices = similarities.argsort()[::-1][:top_k]
        return [self.docs[i] for i in ranked_indices]


def _load_documents(docs_dir: str) -> List[KnowledgeDoc]:
    file_paths = sorted(glob.glob(os.path.join(docs_dir, "*.txt")))
    if not file_paths:
        raise FileNotFoundError(
            f"No knowledge documents found in {docs_dir}. "
            "Add .txt reference notes there before running the pipeline."
        )

    docs = []
    for path in file_paths:
        with open(path, "r", encoding="utf-8") as f:
            docs.append(KnowledgeDoc(source_file=os.path.basename(path), text=f.read()))
    return docs


# ---------------------------------------------------------------------------
# Optional upgrade path: real embeddings + FAISS, matching the diagram exactly.
# Left here (unused by default) so you can swap it in without restructuring
# the rest of the pipeline — it exposes the same `.search()` interface.
# ---------------------------------------------------------------------------
class FaissVectorIndex:
    """
    Same interface as VectorIndex, but backed by sentence embeddings + FAISS.
    Requires: pip install faiss-cpu sentence-transformers
    """

    def __init__(self, docs_dir: str = config.KNOWLEDGE_DOCS_DIR):
        import faiss
        from sentence_transformers import SentenceTransformer

        self.docs = _load_documents(docs_dir)
        self._model = SentenceTransformer("all-MiniLM-L6-v2")

        embeddings = self._model.encode([d.text for d in self.docs])
        dimension = embeddings.shape[1]
        self._index = faiss.IndexFlatL2(dimension)
        self._index.add(embeddings)

    def search(self, query: str, top_k: int = config.TOP_K_RESULTS) -> List[KnowledgeDoc]:
        query_embedding = self._model.encode([query])
        _distances, indices = self._index.search(query_embedding, top_k)
        return [self.docs[i] for i in indices[0]]

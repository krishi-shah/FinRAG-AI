"""
Cross-Encoder Reranker
Reranks retrieved chunks using a cross-encoder model for improved precision.
Cross-encoders jointly encode (query, document) pairs — more accurate than
bi-encoder similarity alone but slower (use as a second stage).
"""

import logging
from typing import List, Dict

logger = logging.getLogger(__name__)


class CrossEncoderReranker:
    """
    Reranks candidate chunks using a cross-encoder model.
    Typical usage: retrieve top-20 with bi-encoder, rerank to top-5 with cross-encoder.
    """

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.model_name = model_name
        self.model = None
        self._load_model()

    def _load_model(self):
        """Load cross-encoder model."""
        try:
            from sentence_transformers import CrossEncoder
            self.model = CrossEncoder(self.model_name)
            logger.info("Cross-encoder loaded: %s", self.model_name)
        except Exception as e:
            logger.warning("Failed to load cross-encoder %s: %s", self.model_name, e)
            self.model = None

    def rerank(self, query: str, chunks: List[Dict], top_k: int = 5) -> List[Dict]:
        """
        Rerank chunks by cross-encoder relevance to the query.

        Args:
            query: User query
            chunks: Candidate chunks (larger set, e.g. top-20 from first stage)
            top_k: Number of results after reranking

        Returns:
            Top-k chunks reranked by cross-encoder score
        """
        if not chunks:
            return []

        if self.model is None:
            logger.warning("Cross-encoder not available, returning original ranking")
            return chunks[:top_k]

        pairs = [(query, chunk["text"]) for chunk in chunks]
        scores = self.model.predict(pairs)

        scored_chunks = []
        for i, chunk in enumerate(chunks):
            reranked = chunk.copy()
            reranked["rerank_score"] = float(scores[i])
            scored_chunks.append(reranked)

        scored_chunks.sort(key=lambda x: x["rerank_score"], reverse=True)

        for i, chunk in enumerate(scored_chunks[:top_k]):
            chunk["rank"] = i + 1

        logger.debug("Reranked %d candidates -> top %d", len(chunks), top_k)
        return scored_chunks[:top_k]

    @property
    def is_available(self) -> bool:
        return self.model is not None

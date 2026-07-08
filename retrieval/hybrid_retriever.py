"""
Hybrid Retriever
Combines BM25 sparse retrieval with FAISS dense retrieval for improved recall.
Uses Reciprocal Rank Fusion (RRF) for score combination.
"""

import logging
from typing import List, Dict, Optional

import numpy as np
from rank_bm25 import BM25Okapi

logger = logging.getLogger(__name__)


class HybridRetriever:
    """
    Combines BM25 (lexical) and dense (semantic) retrieval with configurable weighting.
    
    The alpha parameter controls the balance:
    - alpha=1.0: pure dense retrieval
    - alpha=0.0: pure BM25 retrieval
    - alpha=0.7: default hybrid (70% dense, 30% BM25)
    """

    def __init__(self, alpha: float = 0.7):
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("alpha must be between 0.0 and 1.0")
        self.alpha = alpha
        self.bm25: Optional[BM25Okapi] = None
        self.corpus: List[Dict] = []
        self.tokenized_corpus: List[List[str]] = []
        logger.info("HybridRetriever initialized with alpha=%.2f", alpha)

    def index_documents(self, chunks: List[Dict]):
        """
        Build BM25 index from document chunks.

        Args:
            chunks: List of dicts with at minimum a 'text' field
        """
        self.corpus = chunks
        self.tokenized_corpus = [self._tokenize(chunk["text"]) for chunk in chunks]
        self.bm25 = BM25Okapi(self.tokenized_corpus)
        logger.info("BM25 index built with %d documents", len(chunks))

    def retrieve_bm25(self, query: str, top_k: int = 20) -> List[Dict]:
        """
        Retrieve using BM25 sparse scoring.

        Args:
            query: User query string
            top_k: Number of results to return

        Returns:
            List of chunks with bm25_score
        """
        if self.bm25 is None:
            return []

        tokenized_query = self._tokenize(query)
        scores = self.bm25.get_scores(tokenized_query)

        top_indices = np.argsort(scores)[::-1][:top_k]
        results = []
        for idx in top_indices:
            if scores[idx] > 0:
                chunk = self.corpus[idx].copy()
                chunk["bm25_score"] = float(scores[idx])
                results.append(chunk)

        return results

    def hybrid_search(
        self,
        query: str,
        dense_results: List[Dict],
        top_k: int = 5,
    ) -> List[Dict]:
        """
        Combine dense retrieval results with BM25 scores using Reciprocal Rank Fusion.

        Args:
            query: User query
            dense_results: Results from dense retrieval (must have 'text' and 'similarity_score')
            top_k: Final number of results to return

        Returns:
            Re-scored and merged results
        """
        if self.bm25 is None:
            return dense_results[:top_k]

        bm25_results = self.retrieve_bm25(query, top_k=top_k * 4)

        # RRF scoring
        dense_scores = {}
        for i, r in enumerate(dense_results):
            key = r["text"][:100]
            dense_scores[key] = 1.0 / (i + 1)

        bm25_scores = {}
        for i, r in enumerate(bm25_results):
            key = r["text"][:100]
            bm25_scores[key] = 1.0 / (i + 1)

        # Merge all unique chunks
        all_chunks = {}
        for r in dense_results + bm25_results:
            key = r["text"][:100]
            if key not in all_chunks:
                all_chunks[key] = r.copy()

        # Combine scores
        combined_scores = {}
        for key in all_chunks:
            d_score = dense_scores.get(key, 0.0)
            b_score = bm25_scores.get(key, 0.0)
            combined_scores[key] = self.alpha * d_score + (1 - self.alpha) * b_score

        sorted_keys = sorted(combined_scores.keys(), key=lambda k: combined_scores[k], reverse=True)

        results = []
        for i, key in enumerate(sorted_keys[:top_k]):
            chunk = all_chunks[key].copy()
            chunk["hybrid_score"] = combined_scores[key]
            chunk["dense_rrf"] = dense_scores.get(key, 0.0)
            chunk["bm25_rrf"] = bm25_scores.get(key, 0.0)
            chunk["rank"] = i + 1
            results.append(chunk)

        logger.debug(
            "Hybrid search: %d dense + %d BM25 -> %d combined",
            len(dense_results), len(bm25_results), len(results),
        )
        return results

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        """Simple whitespace tokenization with lowercasing."""
        return text.lower().split()

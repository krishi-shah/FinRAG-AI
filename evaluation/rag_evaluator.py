"""
RAG Evaluation Framework
Evaluates retrieval quality and answer generation using industry-standard metrics:
- Context Recall: Does retrieved context contain answer information?
- Context Precision: Are retrieved chunks relevant to the question?
- Faithfulness: Is the answer grounded in the retrieved context?
- Answer Relevance: Is the answer semantically relevant to the question?
"""

import json
import re
import sys
import time
import logging
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from embeddings.embedder import FinancialEmbedder
from retrieval.rag_pipeline import FinancialRAGPipeline

logger = logging.getLogger(__name__)

GOLDEN_QA_PATH = Path(__file__).parent / "golden_qa.json"
RESULTS_DIR = Path(__file__).parent / "results"


def _tokenize(text: str) -> set:
    """Lowercase word-level tokenization."""
    return set(re.findall(r"\w+", text.lower()))


def _sentence_split(text: str) -> List[str]:
    """Split text into sentences."""
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in sentences if s.strip()]


class RAGEvaluator:
    """Evaluates RAG pipeline quality with multiple metrics."""

    def __init__(self, rag: FinancialRAGPipeline, embedder: FinancialEmbedder):
        self.rag = rag
        self.embedder = embedder

    def context_recall(self, expected_answer: str, context_chunks: List[Dict]) -> float:
        """
        Measures what fraction of the expected answer's key information
        is present in the retrieved context.
        Uses both token overlap and semantic similarity.
        """
        if not expected_answer or not context_chunks:
            return 0.0

        expected_sentences = _sentence_split(expected_answer)
        if not expected_sentences:
            return 0.0

        context_text = " ".join(c.get("text", "") for c in context_chunks)
        context_embedding = self.embedder.generate_embedding(context_text)

        recalled = 0
        for sentence in expected_sentences:
            sent_tokens = _tokenize(sentence)
            ctx_tokens = _tokenize(context_text)
            token_overlap = len(sent_tokens & ctx_tokens) / max(len(sent_tokens), 1)

            sent_embedding = self.embedder.generate_embedding(sentence)
            sim = float(np.dot(sent_embedding, context_embedding))

            if token_overlap > 0.5 or sim > 0.7:
                recalled += 1

        return recalled / len(expected_sentences)

    def context_precision(self, question: str, context_chunks: List[Dict]) -> float:
        """
        Measures what fraction of retrieved chunks are actually relevant to the question.
        """
        if not context_chunks:
            return 0.0

        question_embedding = self.embedder.generate_embedding(question)
        relevant_count = 0

        for chunk in context_chunks:
            chunk_embedding = self.embedder.generate_embedding(chunk.get("text", ""))
            sim = float(np.dot(question_embedding, chunk_embedding))
            if sim > 0.4:
                relevant_count += 1

        return relevant_count / len(context_chunks)

    def faithfulness(self, answer: str, context_chunks: List[Dict]) -> float:
        """
        Measures whether the answer is grounded in the retrieved context.
        """
        if not answer or not context_chunks:
            return 0.0

        answer_sentences = _sentence_split(answer)
        if not answer_sentences:
            return 0.0

        context_text = " ".join(c.get("text", "") for c in context_chunks)
        context_tokens = _tokenize(context_text)
        context_embedding = self.embedder.generate_embedding(context_text)

        grounded = 0
        for sentence in answer_sentences:
            sent_tokens = _tokenize(sentence)
            if not sent_tokens:
                grounded += 1
                continue

            overlap = len(sent_tokens & context_tokens) / len(sent_tokens)
            sent_embedding = self.embedder.generate_embedding(sentence)
            sim = float(np.dot(sent_embedding, context_embedding))

            if overlap > 0.4 or sim > 0.65:
                grounded += 1

        return grounded / len(answer_sentences)

    def answer_relevance(self, question: str, answer: str) -> float:
        """Measures semantic similarity between question and answer."""
        if not question or not answer:
            return 0.0
        q_emb = self.embedder.generate_embedding(question)
        a_emb = self.embedder.generate_embedding(answer)
        sim = float(np.dot(q_emb, a_emb) / (np.linalg.norm(q_emb) * np.linalg.norm(a_emb) + 1e-9))
        return max(0.0, min(1.0, sim))

    def evaluate(self, golden_qa: List[Dict], top_k: int = 3) -> Dict:
        """Run full evaluation on golden QA set."""
        per_question: List[Dict] = []
        total_latency = 0.0

        for qa in golden_qa:
            question = qa["question"]
            expected = qa["expected_answer"]

            start = time.time()
            response = self.rag.query(question, top_k=top_k)
            latency = time.time() - start
            total_latency += latency

            answer = response["answer"]
            sources = response["sources"]

            metrics = {
                "question": question,
                "expected_answer": expected,
                "generated_answer": answer,
                "context_recall": round(self.context_recall(expected, sources), 4),
                "context_precision": round(self.context_precision(question, sources), 4),
                "faithfulness": round(self.faithfulness(answer, sources), 4),
                "answer_relevance": round(self.answer_relevance(question, answer), 4),
                "num_sources": len(sources),
                "latency_ms": round(latency * 1000, 1),
            }
            per_question.append(metrics)

        avg = lambda key: round(float(np.mean([r[key] for r in per_question])), 4)

        summary = {
            "timestamp": datetime.now().isoformat(),
            "num_questions": len(per_question),
            "avg_context_recall": avg("context_recall"),
            "avg_context_precision": avg("context_precision"),
            "avg_faithfulness": avg("faithfulness"),
            "avg_answer_relevance": avg("answer_relevance"),
            "avg_latency_ms": round(total_latency / max(len(per_question), 1) * 1000, 1),
            "per_question": per_question,
        }
        return summary


def run_evaluation(config: Optional[Dict] = None) -> Dict:
    """End-to-end evaluation: build pipeline, load golden set, evaluate, save results."""
    from data_ingestion.sec_downloader import get_sample_sec_data

    pipeline_config = config or {
        "embedding_model": "all-MiniLM-L6-v2",
        "chunk_size": 512,
        "chunk_overlap": 128,
        "top_k": 3,
        "retrieval_mode": "dense",
    }

    rag = FinancialRAGPipeline()
    sample_chunks = get_sample_sec_data()
    embedded = rag.embedder.embed_document_chunks(sample_chunks)
    rag.build_index(embedded)

    with open(GOLDEN_QA_PATH, "r", encoding="utf-8") as f:
        golden_qa = json.load(f)

    evaluator = RAGEvaluator(rag, rag.embedder)
    results = evaluator.evaluate(golden_qa, top_k=pipeline_config.get("top_k", 3))
    results["config"] = pipeline_config

    print("\n" + "=" * 65)
    print("  RAG Evaluation Results")
    print("=" * 65)
    print(f"  Config: chunk_size={pipeline_config['chunk_size']}, overlap={pipeline_config['chunk_overlap']}, mode={pipeline_config['retrieval_mode']}")
    print(f"  Questions evaluated : {results['num_questions']}")
    print(f"  Avg Context Recall  : {results['avg_context_recall']:.4f}")
    print(f"  Avg Context Precision: {results['avg_context_precision']:.4f}")
    print(f"  Avg Faithfulness    : {results['avg_faithfulness']:.4f}")
    print(f"  Avg Answer Relevance: {results['avg_answer_relevance']:.4f}")
    print(f"  Avg Latency         : {results['avg_latency_ms']:.1f}ms")
    print("=" * 65)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = RESULTS_DIR / f"eval_{ts}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {out_path}")

    return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_evaluation()

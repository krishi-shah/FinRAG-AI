"""
RAG Evaluation Framework
<<<<<<< HEAD
Evaluates retrieval quality and answer generation using industry-standard metrics:
- Context Recall: Does retrieved context contain answer information?
- Context Precision: Are retrieved chunks relevant to the question?
- Faithfulness: Is the answer grounded in the retrieved context?
- Answer Relevance: Is the answer semantically relevant to the question?
=======
Computes faithfulness, answer relevance, and context recall metrics
against a golden QA set — no external API required.
>>>>>>> 7d7bc625fee4bf9d4c70c4ee0ef89f65a02aa30c
"""

import json
import re
import sys
<<<<<<< HEAD
import time
import logging
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
=======
import os
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Dict, List
>>>>>>> 7d7bc625fee4bf9d4c70c4ee0ef89f65a02aa30c

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from embeddings.embedder import FinancialEmbedder
from retrieval.rag_pipeline import FinancialRAGPipeline

<<<<<<< HEAD
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
=======
GOLDEN_QA_PATH = Path(__file__).parent / "golden_qa.json"
RESULTS_DIR = Path(__file__).parent / "results"

SAMPLE_CHUNKS = [
    {
        "text": "Apple reported Q4 2023 revenue of $94.8 billion, up 1% year-over-year, driven by strong iPhone sales.",
        "company": "Apple",
        "quarter": "Q4 2023",
        "source": "earnings_call",
        "type": "earnings",
    },
    {
        "text": "Tesla delivered 484,507 vehicles in Q3 2023, exceeding expectations and showing strong demand for electric vehicles.",
        "company": "Tesla",
        "quarter": "Q3 2023",
        "source": "earnings_call",
        "type": "earnings",
    },
    {
        "text": "Microsoft Azure revenue grew 29% year-over-year in the latest quarter, driven by increased cloud adoption.",
        "company": "Microsoft",
        "quarter": "Q3 2023",
        "source": "earnings_call",
        "type": "earnings",
    },
    {
        "text": "Amazon Web Services reported operating income of $7.0 billion, up 12% year-over-year.",
        "company": "Amazon",
        "quarter": "Q3 2023",
        "source": "earnings_call",
        "type": "earnings",
    },
    {
        "text": "The Federal Reserve raised interest rates by 0.25% to combat inflation, affecting tech stock valuations.",
        "company": "Market",
        "quarter": "Q3 2023",
        "source": "news",
        "type": "news",
    },
]


def _tokenize(text: str) -> set:
    """Lowercase word-level tokenization for overlap metrics."""
    return set(re.findall(r"\w+", text.lower()))


class RAGEvaluator:
    """Evaluates a FinancialRAGPipeline on a golden QA set."""
>>>>>>> 7d7bc625fee4bf9d4c70c4ee0ef89f65a02aa30c

    def __init__(self, rag: FinancialRAGPipeline, embedder: FinancialEmbedder):
        self.rag = rag
        self.embedder = embedder

<<<<<<< HEAD
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
=======
    # ------------------------------------------------------------------
    # Metric helpers
    # ------------------------------------------------------------------

    def faithfulness(self, answer: str, context_chunks: List[Dict]) -> float:
        """
        Token-overlap between answer and concatenated retrieved context.
        High overlap => answer is grounded in the context (less hallucination).
        Returns a score in [0, 1].
        """
        answer_tokens = _tokenize(answer)
        if not answer_tokens:
            return 0.0
        context_tokens: set = set()
        for chunk in context_chunks:
            context_tokens |= _tokenize(chunk.get("text", ""))
        overlap = answer_tokens & context_tokens
        return len(overlap) / len(answer_tokens)

    def answer_relevance(self, question: str, answer: str) -> float:
        """
        Cosine similarity between question embedding and answer embedding.
        Measures whether the answer is on-topic.
        Returns a score in [0, 1] (clamped from cosine range).
        """
>>>>>>> 7d7bc625fee4bf9d4c70c4ee0ef89f65a02aa30c
        q_emb = self.embedder.generate_embedding(question)
        a_emb = self.embedder.generate_embedding(answer)
        sim = float(np.dot(q_emb, a_emb) / (np.linalg.norm(q_emb) * np.linalg.norm(a_emb) + 1e-9))
        return max(0.0, min(1.0, sim))

<<<<<<< HEAD
    def evaluate(self, golden_qa: List[Dict], top_k: int = 3) -> Dict:
        """Run full evaluation on golden QA set."""
        per_question: List[Dict] = []
        total_latency = 0.0
=======
    def context_recall(self, expected_answer: str, context_chunks: List[Dict]) -> float:
        """
        Keyword overlap between the *expected* answer and retrieved chunks.
        High recall => the retriever surfaced the right information.
        Returns a score in [0, 1].
        """
        expected_tokens = _tokenize(expected_answer)
        if not expected_tokens:
            return 0.0
        context_tokens: set = set()
        for chunk in context_chunks:
            context_tokens |= _tokenize(chunk.get("text", ""))
        overlap = expected_tokens & context_tokens
        return len(overlap) / len(expected_tokens)

    # ------------------------------------------------------------------
    # Full evaluation run
    # ------------------------------------------------------------------

    def evaluate(self, golden_qa: List[Dict], top_k: int = 3) -> Dict:
        """
        Run all metrics on each golden QA pair and return aggregate results.
        """
        per_question: List[Dict] = []
>>>>>>> 7d7bc625fee4bf9d4c70c4ee0ef89f65a02aa30c

        for qa in golden_qa:
            question = qa["question"]
            expected = qa["expected_answer"]

<<<<<<< HEAD
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
=======
            response = self.rag.query(question, top_k=top_k)
            answer = response["answer"]
            sources = response["sources"]

            faith = self.faithfulness(answer, sources)
            relevance = self.answer_relevance(question, answer)
            recall = self.context_recall(expected, sources)

            per_question.append({
                "question": question,
                "answer": answer,
                "faithfulness": round(faith, 4),
                "answer_relevance": round(relevance, 4),
                "context_recall": round(recall, 4),
                "num_sources": len(sources),
            })

        avg = lambda key: round(np.mean([r[key] for r in per_question]), 4)
        summary = {
            "timestamp": datetime.now().isoformat(),
            "num_questions": len(per_question),
            "avg_faithfulness": avg("faithfulness"),
            "avg_answer_relevance": avg("answer_relevance"),
            "avg_context_recall": avg("context_recall"),
>>>>>>> 7d7bc625fee4bf9d4c70c4ee0ef89f65a02aa30c
            "per_question": per_question,
        }
        return summary


<<<<<<< HEAD
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

=======
def run_evaluation() -> Dict:
    """End-to-end: build pipeline, load golden set, evaluate, save results."""

    # Build RAG pipeline with sample data
    rag = FinancialRAGPipeline()
    embedded = rag.embedder.embed_document_chunks(SAMPLE_CHUNKS)
    rag.build_index(embedded)

    # Load golden QA
>>>>>>> 7d7bc625fee4bf9d4c70c4ee0ef89f65a02aa30c
    with open(GOLDEN_QA_PATH, "r", encoding="utf-8") as f:
        golden_qa = json.load(f)

    evaluator = RAGEvaluator(rag, rag.embedder)
<<<<<<< HEAD
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

=======
    results = evaluator.evaluate(golden_qa, top_k=3)

    # Print summary table
    print("\n" + "=" * 60)
    print("  RAG Evaluation Results")
    print("=" * 60)
    print(f"  Questions evaluated : {results['num_questions']}")
    print(f"  Avg Faithfulness    : {results['avg_faithfulness']:.4f}")
    print(f"  Avg Answer Relevance: {results['avg_answer_relevance']:.4f}")
    print(f"  Avg Context Recall  : {results['avg_context_recall']:.4f}")
    print("=" * 60)

    print("\nPer-question breakdown:")
    for i, r in enumerate(results["per_question"], 1):
        print(f"\n  Q{i}: {r['question']}")
        print(f"      faith={r['faithfulness']:.3f}  rel={r['answer_relevance']:.3f}  recall={r['context_recall']:.3f}")

    # Save to disk
>>>>>>> 7d7bc625fee4bf9d4c70c4ee0ef89f65a02aa30c
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = RESULTS_DIR / f"eval_{ts}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {out_path}")

    return results


if __name__ == "__main__":
<<<<<<< HEAD
    logging.basicConfig(level=logging.INFO)
=======
>>>>>>> 7d7bc625fee4bf9d4c70c4ee0ef89f65a02aa30c
    run_evaluation()

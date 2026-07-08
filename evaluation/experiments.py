"""
Experiment Tracking
Run ablation studies across different RAG configurations and track results.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List

logger = logging.getLogger(__name__)

RESULTS_DIR = Path(__file__).parent / "results"

EXPERIMENT_CONFIGS = [
    {
        "name": "baseline_no_overlap",
        "embedding_model": "all-MiniLM-L6-v2",
        "chunk_size": 1000,
        "chunk_overlap": 0,
        "top_k": 3,
        "retrieval_mode": "dense",
    },
    {
        "name": "optimized_overlap",
        "embedding_model": "all-MiniLM-L6-v2",
        "chunk_size": 512,
        "chunk_overlap": 128,
        "top_k": 5,
        "retrieval_mode": "dense",
    },
    {
        "name": "hybrid_retrieval",
        "embedding_model": "all-MiniLM-L6-v2",
        "chunk_size": 512,
        "chunk_overlap": 128,
        "top_k": 5,
        "retrieval_mode": "hybrid",
    },
    {
        "name": "small_chunks",
        "embedding_model": "all-MiniLM-L6-v2",
        "chunk_size": 256,
        "chunk_overlap": 64,
        "top_k": 7,
        "retrieval_mode": "dense",
    },
]


def run_experiment(config: Dict) -> Dict:
    """Run a single experiment configuration."""
    from evaluation.rag_evaluator import run_evaluation

    logger.info("Running experiment: %s", config.get("name", "unnamed"))
    results = run_evaluation(config)
    results["experiment_name"] = config.get("name", "unnamed")
    return results


def run_all_experiments() -> List[Dict]:
    """Run all predefined experiment configurations and save comparison."""
    all_results = []

    for config in EXPERIMENT_CONFIGS:
        try:
            result = run_experiment(config)
            all_results.append(result)
        except Exception as e:
            logger.error("Experiment '%s' failed: %s", config.get("name"), e)
            all_results.append({
                "experiment_name": config.get("name"),
                "error": str(e),
                "config": config,
            })

    summary = generate_comparison(all_results)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = RESULTS_DIR / f"experiments_{ts}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\nExperiment results saved to {out_path}")
    return all_results


def generate_comparison(results: List[Dict]) -> Dict:
    """Generate a comparison table from experiment results."""
    comparison = {
        "timestamp": datetime.now().isoformat(),
        "num_experiments": len(results),
        "experiments": [],
    }

    print("\n" + "=" * 80)
    print("  EXPERIMENT COMPARISON")
    print("=" * 80)
    print(f"  {'Experiment':<25} {'Recall':>8} {'Precision':>10} {'Faithful':>10} {'Relevance':>10} {'Latency':>10}")
    print("  " + "-" * 73)

    for r in results:
        if "error" in r:
            print(f"  {r['experiment_name']:<25} {'ERROR':>8}")
            comparison["experiments"].append({"name": r["experiment_name"], "error": r["error"]})
            continue

        row = {
            "name": r.get("experiment_name", "?"),
            "config": r.get("config", {}),
            "context_recall": r.get("avg_context_recall", 0),
            "context_precision": r.get("avg_context_precision", 0),
            "faithfulness": r.get("avg_faithfulness", 0),
            "answer_relevance": r.get("avg_answer_relevance", 0),
            "latency_ms": r.get("avg_latency_ms", 0),
        }
        comparison["experiments"].append(row)

        print(
            f"  {row['name']:<25} {row['context_recall']:>8.4f} "
            f"{row['context_precision']:>10.4f} {row['faithfulness']:>10.4f} "
            f"{row['answer_relevance']:>10.4f} {row['latency_ms']:>8.1f}ms"
        )

    print("=" * 80)

    valid = [e for e in comparison["experiments"] if "error" not in e]
    if valid:
        best = max(valid, key=lambda x: x["context_recall"])
        comparison["best_config"] = best["name"]
        comparison["best_context_recall"] = best["context_recall"]
        print(f"\n  Best config: {best['name']} (context_recall={best['context_recall']:.4f})")

    return comparison


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_all_experiments()

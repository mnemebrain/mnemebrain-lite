"""Belief Maintenance Benchmark (BMB) -- CLI entry point.

Runs 30 tasks across 5 categories against memory system adapters:
  - Contradiction detection (6 tasks)
  - Belief revision (6 tasks)
  - Evidence tracking (6 tasks)
  - Temporal updates (6 tasks)
  - Counterfactual reasoning (6 tasks)

Usage:
    python -m mnemebrain_core.benchmark.bmb_cli
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from mnemebrain_core.benchmark.interface import MemorySystem
from mnemebrain_core.benchmark.scenarios.loader import load_scenarios
from mnemebrain_core.benchmark.system_runner import SystemBenchmarkRunner
from mnemebrain_core.benchmark.system_report import format_scorecard, export_json

_BMB_SCENARIOS_PATH = Path(__file__).parent / "scenarios" / "data" / "bmb_scenarios.json"

BMB_CATEGORIES = [
    "contradiction",
    "belief_revision",
    "evidence_tracking",
    "temporal",
    "counterfactual",
]

ALL_ADAPTERS = [
    "naive_baseline",
    "langchain_buffer",
    "rag_baseline",
    "structured_memory",
]


def _build_adapters(adapter_filter: str | None = None) -> list[MemorySystem]:
    """Build adapters for the BMB benchmark."""
    adapters: list[MemorySystem] = []

    embedder = None
    def _get_embedder():
        nonlocal embedder
        if embedder is None:
            from mnemebrain_core.providers.embeddings.sentence_transformers import (
                SentenceTransformerProvider,
            )
            embedder = SentenceTransformerProvider()
        return embedder

    if adapter_filter is None or adapter_filter == "naive_baseline":
        try:
            from mnemebrain_core.benchmark.adapters.naive_baseline import NaiveBaseline
            adapters.append(NaiveBaseline(_get_embedder()))
        except ImportError:
            if adapter_filter == "naive_baseline":
                print("naive_baseline requires sentence-transformers: pip install mnemebrain-lite[embeddings]")
                sys.exit(1)

    if adapter_filter is None or adapter_filter == "langchain_buffer":
        from mnemebrain_core.benchmark.adapters.langchain_buffer import LangChainBufferBaseline
        adapters.append(LangChainBufferBaseline())

    if adapter_filter is None or adapter_filter == "rag_baseline":
        try:
            from mnemebrain_core.benchmark.adapters.rag_baseline import RAGBaseline
            adapters.append(RAGBaseline(_get_embedder()))
        except ImportError:
            if adapter_filter == "rag_baseline":
                print("rag_baseline requires sentence-transformers: pip install mnemebrain-lite[embeddings]")
                sys.exit(1)

    if adapter_filter is None or adapter_filter == "structured_memory":
        try:
            from mnemebrain_core.benchmark.adapters.structured_memory import StructuredMemoryBaseline
            adapters.append(StructuredMemoryBaseline(_get_embedder()))
        except ImportError:
            if adapter_filter == "structured_memory":
                print("structured_memory requires sentence-transformers: pip install mnemebrain-lite[embeddings]")
                sys.exit(1)

    return adapters


def _print_bmb_chart(results: dict[str, list]) -> None:
    """Print the BMB bar chart."""
    from mnemebrain_core.benchmark.scoring import aggregate_by_category

    print("\n" + "=" * 60)
    print("  BELIEF MAINTENANCE BENCHMARK (BMB)")
    print("  30 tasks | 5 categories | 90 max points")
    print("=" * 60)

    for system_name, scores in results.items():
        cats = aggregate_by_category(scores)
        scored = [c for c in cats.values() if not c.skipped and c.score is not None]
        if scored:
            avg = sum(c.score for c in scored) / len(scored)
            pct = int(avg * 100)
            bar = "\u2588" * (pct // 5)
            print(f"  {system_name:<20} {bar} {pct}%")
        else:
            print(f"  {system_name:<20} N/A")

    print("=" * 60)
    print()


def run_bmb(
    adapter_filter: str | None = None,
    category: str | None = None,
    scenario_name: str | None = None,
    output: str = "bmb_report.json",
) -> dict[str, list]:
    """Run the BMB benchmark and return results."""
    scenarios = load_scenarios(_BMB_SCENARIOS_PATH)

    if category:
        scenarios = [s for s in scenarios if s.category == category]
    if scenario_name:
        scenarios = [s for s in scenarios if s.name == scenario_name]

    if not scenarios:
        print("No matching BMB scenarios found.")
        sys.exit(1)

    adapters = _build_adapters(adapter_filter)
    if not adapters:
        print("No matching adapters found.")
        sys.exit(1)

    print(f"BMB: Running {len(scenarios)} scenarios against {len(adapters)} adapter(s)...\n")

    runner = SystemBenchmarkRunner()
    results = runner.run_all(adapters, scenarios)

    print(format_scorecard(results))
    _print_bmb_chart(results)
    export_json(results, output)
    print(f"Report saved to {output}")

    return results


def main(argv: list[str] | None = None) -> None:
    """CLI entry point for the BMB benchmark."""
    parser = argparse.ArgumentParser(
        description="MnemeBrain Belief Maintenance Benchmark (BMB)"
    )
    parser.add_argument(
        "--adapter", type=str, default=None,
        choices=ALL_ADAPTERS,
        help="Run only a specific adapter",
    )
    parser.add_argument(
        "--category", type=str, default=None,
        choices=BMB_CATEGORIES,
        help="Run only scenarios in a specific BMB category",
    )
    parser.add_argument(
        "--scenario", type=str, default=None,
        help="Run only a specific scenario by name",
    )
    parser.add_argument(
        "--output", type=str, default="bmb_report.json",
        help="Output path for JSON report (default: bmb_report.json)",
    )

    args = parser.parse_args(argv)
    run_bmb(
        adapter_filter=args.adapter,
        category=args.category,
        scenario_name=args.scenario,
        output=args.output,
    )


if __name__ == "__main__":
    main()

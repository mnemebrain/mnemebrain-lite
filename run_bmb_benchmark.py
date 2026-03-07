#!/usr/bin/env python
"""Run the Belief Maintenance Benchmark (BMB).

Usage:
    python run_bmb_benchmark.py
    python run_bmb_benchmark.py --adapter naive_baseline
    python run_bmb_benchmark.py --category contradiction
"""
from mnemebrain_core.benchmark.bmb_cli import main

if __name__ == "__main__":
    main()

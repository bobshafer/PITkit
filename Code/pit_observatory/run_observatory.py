#!/usr/bin/env python3
"""
run_observatory.py
------------------

Master orchestrator for the PIT Automated Observatory experiment.
This script sequentially runs:
    1. pit_simulator.py — generates initial Φ–K dynamics data
    2. sweep.py         — runs parameter sweeps and logs metrics
    3. analyze.py       — interprets data and identifies coherence zones
    4. visualize.py     — produces visual summaries

Usage:
    python run_observatory.py
"""

import subprocess
import sys
import time
from pathlib import Path

MODULES = [
    ("pit_simulator.py", "Generating initial Φ–K evolution data..."),
    ("sweep.py", "Running parameter sweeps across μ–ν space..."),
    ("analyze.py", "Analyzing data for coherence signatures..."),
    ("visualize.py", "Visualizing coherence landscape...")
]

LOG_FILE = Path("observatory.log")

def run_module(script, message):
    """Run a single module as a subprocess."""
    print(f"\n🧩 {message}")
    print(f"→ Executing {script} ...")
    start = time.time()

    result = subprocess.run([sys.executable, script], capture_output=True, text=True)
    duration = time.time() - start

    log_entry = f"\n--- {script} ({duration:.2f}s) ---\n{result.stdout}\n{result.stderr}\n"
    LOG_FILE.write_text(LOG_FILE.read_text() + log_entry if LOG_FILE.exists() else log_entry)

    if result.returncode != 0:
        print(f"❌ {script} failed. Check observatory.log for details.")
        sys.exit(result.returncode)
    else:
        print(f"✅ {script} completed successfully ({duration:.2f}s).")

def main():
    print("🌌 Starting PIT Automated Observatory sequence...")
    LOG_FILE.write_text(f"=== PIT Observatory Run {time.ctime()} ===\n")

    for script, message in MODULES:
        if not Path(script).exists():
            print(f"⚠️  Missing {script} — skipping.")
            continue
        run_module(script, message)

    print("\n🌠 Observatory sequence complete!")
    print(f"📄 Logs written to: {LOG_FILE.resolve()}")

if __name__ == "__main__":
    main()


#!/usr/bin/env bash
set -euo pipefail

echo "== load_kb =="
python scripts/load_kb.py

echo "== generate_messages =="
python scripts/generate_messages.py

echo "== run_triage =="
python scripts/run_triage.py

echo "== channel_triage_summary =="
python scripts/channel_triage_summary.py

echo "== evaluate_llm =="
python scripts/evaluate_llm.py

#!/usr/bin/env bash
set -euo pipefail
shopt -s nullglob
for s in *; do
  [[ "$s" =~ ^[0-9]+$ ]] || continue
  echo "Submitting $s"
  sbatch "$s"
done

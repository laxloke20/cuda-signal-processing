#!/usr/bin/env bash
set -euo pipefail

mkdir -p output

make clean
make

./signal_processing.exe \
  --signals 256 \
  --samples 8192 \
  --window 9 \
  --output output/processed_samples.csv \
  | tee output/execution.log

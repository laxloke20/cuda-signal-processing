# CUDA Batch Signal Denoising and Feature Extraction

This project demonstrates large-scale GPU signal processing with CUDA. It generates a batch of independent noisy synthetic signals, sends the dataset to the GPU, applies a moving-average denoising filter, and then computes the first difference of the filtered signal as a simple feature-extraction stage.

The default run processes **256 signals with 8,192 samples each**, or **2,097,152 samples in one execution**.

## Why this project

The goal is to demonstrate a workload that is large enough to benefit from GPU parallelism while remaining easy to reproduce. Each CUDA thread processes a signal sample. Signal boundaries are handled explicitly so that the moving-average window never reads samples belonging to a neighboring signal.

## GPU computation

Two CUDA kernels are used:

1. **MovingAverageKernel** – applies an odd-width moving-average/FIR-style smoothing window to remove high-frequency noise.
2. **DifferenceKernel** – calculates the sample-to-sample difference of the filtered signal to highlight rapid changes.

The project also compares a subset of GPU-filtered values against a CPU reference implementation and reports the maximum absolute error.

## Requirements

- NVIDIA GPU with CUDA support
- CUDA Toolkit with `nvcc`
- C++17-capable CUDA compiler
- Linux shell for `run.sh`

## Build

```bash
make
```

## Run

```bash
./signal_processing.exe --signals 256 --samples 8192 --window 9
```

Available CLI arguments:

```text
--signals N     Number of independent signals
--samples N     Samples per signal
--window N      Odd moving-average window size
--output PATH   CSV output path
--help          Show usage
```

For example, a larger run is:

```bash
./signal_processing.exe \
  --signals 512 \
  --samples 16384 \
  --window 9
```

## Reproducible course run

```bash
chmod +x run.sh
./run.sh
```

`run.sh` rebuilds the project and executes the default large-data configuration. The console output is saved to:

```text
output/execution.log
```

A representative subset of processed samples is saved to:

```text
output/processed_samples.csv
```

The CSV intentionally contains only a small representative subset so the repository does not need to store millions of rows. The execution log records the total number of samples processed in the full GPU workload.

## Default workload

- Signals: 256
- Samples per signal: 8,192
- Total samples: 2,097,152
- Filter window: 9 samples
- CUDA threads per block: 256

## Expected output

A successful run prints information similar to:

```text
CUDA Batch Signal Processing Project
GPU: NVIDIA L4
Signals: 256
Samples per signal: 8192
Total samples: 2097152
Moving-average window: 9

Generating synthetic noisy signal dataset...
Copying 2097152 samples to GPU...
Running moving-average CUDA kernel...
Running signal-difference CUDA kernel...

GPU processing completed.
Kernel time: ...
Total samples processed: 2097152
CPU/GPU validation max error: ...
Results saved to: output/processed_samples.csv
```

Exact timing depends on the GPU.

## Dataset

The dataset is generated deterministically in the program using a fixed random seed. Each signal combines:

- a sine wave,
- a signal-dependent frequency and phase, and
- Gaussian noise.

This makes the project fully reproducible and removes the need to download external data while still satisfying the requirement to process hundreds of signal inputs.

## Validation

The program computes the moving-average result on the CPU for the first four complete signals and compares those values with the GPU result. It prints the maximum absolute error as a correctness check.

## Lessons learned

The main challenge was preventing the filter window from crossing signal boundaries when many signals were stored in one flattened GPU array. Each CUDA thread therefore calculates both its global index and its sample position within the current signal before choosing the filter range.

The project also demonstrates that GPU kernels are most useful when the workload contains a large amount of independent data-parallel work. Flattening hundreds of signals into one large array allows millions of samples to be processed with the same CUDA launch.

## Repository contents

```text
signal_processing.cu         CUDA kernels and host application
Makefile                     Build and clean targets
run.sh                       Reproducible build/run script
output/execution.log         Proof of execution after running
output/processed_samples.csv Representative processed values
screenshots/                 Optional screenshots of execution
```

## Course submission description

This project implements a CUDA-based batch signal-processing pipeline. It processes hundreds of independent noisy signals in a single execution. A moving-average CUDA kernel performs denoising, and a second CUDA kernel calculates sample-to-sample differences for feature extraction. The default configuration processes more than two million signal samples. The program includes command-line configuration, CUDA event timing, CPU/GPU correctness validation, CSV output, a Makefile, and a reproducible run script.

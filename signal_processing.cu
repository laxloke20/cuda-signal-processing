#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr float kPi = 3.14159265358979323846f;

struct Options {
  int signals = 256;
  int samples = 8192;
  int window = 9;
  std::string output_path = "output/processed_samples.csv";
};

void CheckCuda(cudaError_t error, const char* expression, const char* file,
               int line) {
  if (error != cudaSuccess) {
    std::cerr << "CUDA error at " << file << ":" << line << " for "
              << expression << ": " << cudaGetErrorString(error) << "\n";
    std::exit(EXIT_FAILURE);
  }
}

#define CUDA_CHECK(call) CheckCuda((call), #call, __FILE__, __LINE__)

void PrintUsage(const char* executable) {
  std::cout
      << "Usage: " << executable
      << " [--signals N] [--samples N] [--window ODD_N] [--output PATH]\n";
}

int ParsePositiveInt(const std::string& value, const std::string& option) {
  const int parsed = std::stoi(value);
  if (parsed <= 0) {
    throw std::invalid_argument(option + " must be greater than zero");
  }
  return parsed;
}

Options ParseArguments(int argc, char** argv) {
  Options options;

  for (int i = 1; i < argc; ++i) {
    const std::string argument = argv[i];

    auto RequireValue = [&](const std::string& option) -> std::string {
      if (i + 1 >= argc) {
        throw std::invalid_argument("Missing value for " + option);
      }
      return argv[++i];
    };

    if (argument == "--signals") {
      options.signals =
          ParsePositiveInt(RequireValue(argument), argument);
    } else if (argument == "--samples") {
      options.samples =
          ParsePositiveInt(RequireValue(argument), argument);
    } else if (argument == "--window") {
      options.window =
          ParsePositiveInt(RequireValue(argument), argument);
    } else if (argument == "--output") {
      options.output_path = RequireValue(argument);
    } else if (argument == "--help" || argument == "-h") {
      PrintUsage(argv[0]);
      std::exit(EXIT_SUCCESS);
    } else {
      throw std::invalid_argument("Unknown argument: " + argument);
    }
  }

  if (options.window % 2 == 0) {
    throw std::invalid_argument("--window must be an odd number");
  }

  return options;
}

std::vector<float> GenerateSignals(int signal_count, int samples_per_signal) {
  const std::size_t total_samples =
      static_cast<std::size_t>(signal_count) * samples_per_signal;

  std::vector<float> input(total_samples);
  std::mt19937 generator(42);
  std::normal_distribution<float> noise(0.0f, 0.30f);

  for (int signal = 0; signal < signal_count; ++signal) {
    const float frequency = 2.0f + static_cast<float>(signal % 11);
    const float phase = 0.07f * static_cast<float>(signal % 17);
    const std::size_t base =
        static_cast<std::size_t>(signal) * samples_per_signal;

    for (int sample = 0; sample < samples_per_signal; ++sample) {
      const float time =
          static_cast<float>(sample) / static_cast<float>(samples_per_signal);
      const float clean =
          std::sin(2.0f * kPi * frequency * time + phase);
      input[base + sample] = clean + noise(generator);
    }
  }

  return input;
}

__global__ void MovingAverageKernel(const float* input, float* filtered,
                                    int signal_count,
                                    int samples_per_signal, int half_window) {
  const std::size_t index =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const std::size_t total_samples =
      static_cast<std::size_t>(signal_count) * samples_per_signal;

  if (index >= total_samples) {
    return;
  }

  const int sample =
      static_cast<int>(index % static_cast<std::size_t>(samples_per_signal));
  const std::size_t signal_base =
      index - static_cast<std::size_t>(sample);

  const int begin = max(0, sample - half_window);
  const int end = min(samples_per_signal - 1, sample + half_window);

  float sum = 0.0f;
  int count = 0;
  for (int current = begin; current <= end; ++current) {
    sum += input[signal_base + current];
    ++count;
  }

  filtered[index] = sum / static_cast<float>(count);
}

__global__ void DifferenceKernel(const float* filtered, float* difference,
                                 int signal_count, int samples_per_signal) {
  const std::size_t index =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const std::size_t total_samples =
      static_cast<std::size_t>(signal_count) * samples_per_signal;

  if (index >= total_samples) {
    return;
  }

  const int sample =
      static_cast<int>(index % static_cast<std::size_t>(samples_per_signal));

  if (sample == 0) {
    difference[index] = 0.0f;
  } else {
    difference[index] = filtered[index] - filtered[index - 1];
  }
}

float ValidateSubset(const std::vector<float>& input,
                     const std::vector<float>& gpu_filtered,
                     int signal_count, int samples_per_signal, int window) {
  const int signals_to_validate = std::min(signal_count, 4);
  const int half_window = window / 2;
  float maximum_error = 0.0f;

  for (int signal = 0; signal < signals_to_validate; ++signal) {
    const std::size_t base =
        static_cast<std::size_t>(signal) * samples_per_signal;

    for (int sample = 0; sample < samples_per_signal; ++sample) {
      const int begin = std::max(0, sample - half_window);
      const int end =
          std::min(samples_per_signal - 1, sample + half_window);

      float sum = 0.0f;
      int count = 0;
      for (int current = begin; current <= end; ++current) {
        sum += input[base + current];
        ++count;
      }

      const float cpu_value = sum / static_cast<float>(count);
      const float error =
          std::fabs(cpu_value - gpu_filtered[base + sample]);
      maximum_error = std::max(maximum_error, error);
    }
  }

  return maximum_error;
}

void SaveSampleCsv(const std::string& path, const std::vector<float>& input,
                   const std::vector<float>& filtered,
                   const std::vector<float>& difference, int signal_count,
                   int samples_per_signal) {
  const std::filesystem::path output_path(path);
  if (output_path.has_parent_path()) {
    std::filesystem::create_directories(output_path.parent_path());
  }

  std::ofstream output(path);
  if (!output) {
    throw std::runtime_error("Unable to open output file: " + path);
  }

  output << "signal,sample,input,filtered,difference\n";
  output << std::fixed << std::setprecision(6);

  const int signals_to_save = std::min(signal_count, 8);
  const int samples_to_save = std::min(samples_per_signal, 64);

  for (int signal = 0; signal < signals_to_save; ++signal) {
    const std::size_t base =
        static_cast<std::size_t>(signal) * samples_per_signal;

    for (int sample = 0; sample < samples_to_save; ++sample) {
      const std::size_t index = base + sample;
      output << signal << "," << sample << "," << input[index] << ","
             << filtered[index] << "," << difference[index] << "\n";
    }
  }
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const Options options = ParseArguments(argc, argv);
    const std::size_t total_samples =
        static_cast<std::size_t>(options.signals) * options.samples;
    const std::size_t bytes = total_samples * sizeof(float);

    int device = 0;
    CUDA_CHECK(cudaGetDevice(&device));

    cudaDeviceProp properties{};
    CUDA_CHECK(cudaGetDeviceProperties(&properties, device));

    std::cout << "CUDA Batch Signal Processing Project\n";
    std::cout << "GPU: " << properties.name << "\n";
    std::cout << "Signals: " << options.signals << "\n";
    std::cout << "Samples per signal: " << options.samples << "\n";
    std::cout << "Total samples: " << total_samples << "\n";
    std::cout << "Moving-average window: " << options.window << "\n";
    std::cout << "Input size: " << std::fixed << std::setprecision(2)
              << static_cast<double>(bytes) / (1024.0 * 1024.0) << " MiB\n\n";

    std::cout << "Generating synthetic noisy signal dataset...\n";
    const std::vector<float> input =
        GenerateSignals(options.signals, options.samples);

    std::vector<float> filtered(total_samples);
    std::vector<float> difference(total_samples);

    float* device_input = nullptr;
    float* device_filtered = nullptr;
    float* device_difference = nullptr;

    CUDA_CHECK(cudaMalloc(&device_input, bytes));
    CUDA_CHECK(cudaMalloc(&device_filtered, bytes));
    CUDA_CHECK(cudaMalloc(&device_difference, bytes));

    std::cout << "Copying " << total_samples << " samples to GPU...\n";
    CUDA_CHECK(cudaMemcpy(device_input, input.data(), bytes,
                          cudaMemcpyHostToDevice));

    constexpr int kThreadsPerBlock = 256;
    const int blocks = static_cast<int>(
        (total_samples + kThreadsPerBlock - 1) / kThreadsPerBlock);

    cudaEvent_t start;
    cudaEvent_t stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    std::cout << "Running moving-average CUDA kernel...\n";
    std::cout << "Running signal-difference CUDA kernel...\n";

    CUDA_CHECK(cudaEventRecord(start));

    MovingAverageKernel<<<blocks, kThreadsPerBlock>>>(
        device_input, device_filtered, options.signals, options.samples,
        options.window / 2);
    CUDA_CHECK(cudaGetLastError());

    DifferenceKernel<<<blocks, kThreadsPerBlock>>>(
        device_filtered, device_difference, options.signals, options.samples);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float kernel_milliseconds = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&kernel_milliseconds, start, stop));

    CUDA_CHECK(cudaMemcpy(filtered.data(), device_filtered, bytes,
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(difference.data(), device_difference, bytes,
                          cudaMemcpyDeviceToHost));

    const float maximum_error =
        ValidateSubset(input, filtered, options.signals, options.samples,
                       options.window);

    SaveSampleCsv(options.output_path, input, filtered, difference,
                  options.signals, options.samples);

    std::cout << "\nGPU processing completed.\n";
    std::cout << "Kernel time: " << std::fixed << std::setprecision(3)
              << kernel_milliseconds << " ms\n";
    std::cout << "Total samples processed: " << total_samples << "\n";
    std::cout << "CPU/GPU validation max error: " << std::scientific
              << maximum_error << "\n";
    std::cout << "Results saved to: " << options.output_path << "\n";

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(device_input));
    CUDA_CHECK(cudaFree(device_filtered));
    CUDA_CHECK(cudaFree(device_difference));

    return EXIT_SUCCESS;
  } catch (const std::exception& error) {
    std::cerr << "Error: " << error.what() << "\n";
    PrintUsage(argv[0]);
    return EXIT_FAILURE;
  }
}

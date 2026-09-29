#include <cuda_runtime.h>

#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

namespace {

constexpr int kBoardSize = 9;
constexpr int kThreads = 32;

void CheckCuda(cudaError_t error, const char* expression, const char* file,
               int line) {
  if (error != cudaSuccess) {
    std::cerr << "CUDA error at " << file << ":" << line << " for "
              << expression << ": " << cudaGetErrorString(error) << "\n";
    std::exit(EXIT_FAILURE);
  }
}

#define CUDA_CHECK(call) CheckCuda((call), #call, __FILE__, __LINE__)

__device__ bool DeviceWins(const int* board, int player) {
  const int lines[8][3] = {
      {0, 1, 2}, {3, 4, 5}, {6, 7, 8}, {0, 3, 6},
      {1, 4, 7}, {2, 5, 8}, {0, 4, 8}, {2, 4, 6}};

  for (int i = 0; i < 8; ++i) {
    if (board[lines[i][0]] == player &&
        board[lines[i][1]] == player &&
        board[lines[i][2]] == player) {
      return true;
    }
  }
  return false;
}

__global__ void ScoreMovesKernel(const int* board, int player, int* scores) {
  const int cell = threadIdx.x;
  if (cell >= kBoardSize) {
    return;
  }

  if (board[cell] != 0) {
    scores[cell] = -1000000;
    return;
  }

  int trial[kBoardSize];
  for (int i = 0; i < kBoardSize; ++i) {
    trial[i] = board[i];
  }

  trial[cell] = player;

  if (DeviceWins(trial, player)) {
    scores[cell] = 1000;
    return;
  }

  const int opponent = (player == 1) ? 2 : 1;
  trial[cell] = opponent;
  if (DeviceWins(trial, opponent)) {
    scores[cell] = 800;
    return;
  }

  if (cell == 4) {
    scores[cell] = 500;
  } else if (cell == 0 || cell == 2 || cell == 6 || cell == 8) {
    scores[cell] = 300;
  } else {
    scores[cell] = 100;
  }

  // Deterministic tie-breaking gives each competitor a slightly
  // different preference while keeping the strategy understandable.
  if (player == 1) {
    scores[cell] += (8 - cell);
  } else {
    scores[cell] += cell;
  }
}

bool HostWins(const int* board, int player) {
  const int lines[8][3] = {
      {0, 1, 2}, {3, 4, 5}, {6, 7, 8}, {0, 3, 6},
      {1, 4, 7}, {2, 5, 8}, {0, 4, 8}, {2, 4, 6}};

  for (const auto& line : lines) {
    if (board[line[0]] == player && board[line[1]] == player &&
        board[line[2]] == player) {
      return true;
    }
  }
  return false;
}

bool BoardFull(const int* board) {
  for (int i = 0; i < kBoardSize; ++i) {
    if (board[i] == 0) {
      return false;
    }
  }
  return true;
}

char Symbol(int value) {
  if (value == 1) return 'X';
  if (value == 2) return 'O';
  return ' ';
}

void PrintBoard(const int* board) {
  std::cout << "\n";
  for (int row = 0; row < 3; ++row) {
    const int base = row * 3;
    std::cout << " " << Symbol(board[base]) << " | "
              << Symbol(board[base + 1]) << " | "
              << Symbol(board[base + 2]) << "\n";
    if (row != 2) {
      std::cout << "---+---+---\n";
    }
  }
  std::cout << "\n";
}

int ChooseMoveOnGpu(const int* host_board, int player, int physical_device) {
  CUDA_CHECK(cudaSetDevice(physical_device));

  int* device_board = nullptr;
  int* device_scores = nullptr;
  CUDA_CHECK(cudaMalloc(&device_board, kBoardSize * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&device_scores, kBoardSize * sizeof(int)));

  CUDA_CHECK(cudaMemcpy(device_board, host_board, kBoardSize * sizeof(int),
                        cudaMemcpyHostToDevice));

  ScoreMovesKernel<<<1, kThreads>>>(device_board, player, device_scores);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  int scores[kBoardSize];
  CUDA_CHECK(cudaMemcpy(scores, device_scores, kBoardSize * sizeof(int),
                        cudaMemcpyDeviceToHost));

  CUDA_CHECK(cudaFree(device_board));
  CUDA_CHECK(cudaFree(device_scores));

  int best_cell = -1;
  int best_score = -1000001;
  for (int cell = 0; cell < kBoardSize; ++cell) {
    if (scores[cell] > best_score) {
      best_score = scores[cell];
      best_cell = cell;
    }
  }
  return best_cell;
}

}  // namespace

int main() {
  int device_count = 0;
  CUDA_CHECK(cudaGetDeviceCount(&device_count));

  if (device_count < 1) {
    std::cerr << "No CUDA-capable GPU found.\n";
    return EXIT_FAILURE;
  }

  std::cout << "CUDA GPU Tic-Tac-Toe\n";
  std::cout << "Detected physical GPUs: " << device_count << "\n";

  if (device_count >= 2) {
    std::cout << "GPU competitor X -> physical GPU 0\n";
    std::cout << "GPU competitor O -> physical GPU 1\n";
  } else {
    std::cout << "Single-GPU environment detected.\n";
    std::cout << "Both logical GPU competitors will run sequentially on "
                 "physical GPU 0 for demonstration.\n";
  }

  int board[kBoardSize] = {0};
  PrintBoard(board);

  int player = 1;
  int turn = 1;

  while (true) {
    const int logical_gpu = (player == 1) ? 0 : 1;
    const int physical_gpu =
        (device_count >= 2) ? logical_gpu : 0;

    const int move = ChooseMoveOnGpu(board, player, physical_gpu);
    if (move < 0) {
      std::cout << "No valid move available.\n";
      break;
    }

    board[move] = player;

    std::cout << "Turn " << turn << ": logical GPU " << logical_gpu
              << " (" << (player == 1 ? 'X' : 'O') << ") chose cell "
              << move << " using physical GPU " << physical_gpu << ".\n";
    PrintBoard(board);

    if (HostWins(board, player)) {
      std::cout << "Winner: logical GPU " << logical_gpu << " ("
                << (player == 1 ? 'X' : 'O') << ")\n";
      break;
    }

    if (BoardFull(board)) {
      std::cout << "Result: draw\n";
      break;
    }

    player = (player == 1) ? 2 : 1;
    ++turn;
  }

  return EXIT_SUCCESS;
}

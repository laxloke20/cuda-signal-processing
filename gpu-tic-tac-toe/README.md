# CUDA GPU Tic-Tac-Toe

## Project description

This project designs a simple Tic-Tac-Toe game in which two GPU competitors take turns choosing moves. Logical **GPU 0** plays X and logical **GPU 1** plays O. Each competitor launches a CUDA kernel that scores every available square in parallel and then selects the highest-scoring move.

When two physical CUDA devices are available, X runs on physical GPU 0 and O runs on physical GPU 1. When only one GPU is available, the program keeps the same two-competitor design but executes both logical competitors sequentially on physical GPU 0 so that the design can still be demonstrated in a single-GPU environment such as Google Colab.

## Move strategy

For every empty square, a CUDA thread creates a trial board and assigns a score:

- 1000: the move immediately wins
- 800: the move blocks an immediate opponent win
- 500: take the center
- 300: take a corner
- 100: take an edge

A small deterministic tie-break is added so the two competitors do not always prefer identical cells.

## Game flow

1. Start with an empty 3 × 3 board.
2. X asks its GPU kernel to score all legal moves.
3. X selects the highest-scoring cell.
4. The board is printed.
5. O repeats the same process with its GPU competitor.
6. Turns continue until X wins, O wins, or the board is full.

## Build

```bash
nvcc -std=c++17 gpu_tic_tac_toe.cu -o gpu_tic_tac_toe
```

## Run

```bash
./gpu_tic_tac_toe
```

## Visualization

The terminal prints the board after every move, for example:

```text
Turn 1: logical GPU 0 (X) chose cell 4

   |   |
---+---+---
   | X |
---+---+---
   |   |

Turn 2: logical GPU 1 (O) chose cell 8

   |   |
---+---+---
   | X |
---+---+---
   |   | O
```

The purpose of the project is to demonstrate how two GPU-based competitors can evaluate possible actions and alternate turns while sharing a small game state.

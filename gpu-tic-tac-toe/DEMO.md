# GPU Tic-Tac-Toe — Game Visualization

The game uses the standard 3 × 3 Tic-Tac-Toe board.

## Competitors

- **Logical GPU 0 = X**
- **Logical GPU 1 = O**

Each turn, the active competitor launches a CUDA kernel with one thread assigned to each possible board cell. The threads independently score legal moves. The host selects the highest-scoring move and updates the shared board.

## Example turn-by-turn visualization

### Initial board

```text
   |   |
---+---+---
   |   |
---+---+---
   |   |
```

### GPU 0 chooses the center

```text
   |   |
---+---+---
   | X |
---+---+---
   |   |
```

### GPU 1 chooses a corner

```text
   |   |
---+---+---
   | X |
---+---+---
   |   | O
```

### GPU 0 chooses another corner

```text
 X |   |
---+---+---
   | X |
---+---+---
   |   | O
```

The actual program prints the board after every move until there is a winner or a draw.

## How a move is selected

```text
Current board
     |
     v
Copy board to active GPU
     |
     v
CUDA kernel scores all 9 cells in parallel
     |
     +--> winning move       = 1000
     +--> block opponent     = 800
     +--> center             = 500
     +--> corner             = 300
     +--> edge               = 100
     |
     v
Copy scores back to host
     |
     v
Choose highest score
     |
     v
Update + print board
```

With two physical GPUs, each competitor uses a different CUDA device. In a single-GPU environment, both logical competitors execute sequentially on the available GPU while preserving the same game logic and turn structure.

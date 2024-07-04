# juliaNN

This is a neural network package implemented in Julia so I could better understand both the language and artificial neural networks.

### Files
| Filename | Description |
|----------|-------------|
| `autoincoder.jl`                | Generates a 8x3x8 autoencoder  |
| `CANN_utils.jl`                 | Some neural network utility functions (read/write/encode/decode, etc..) |
| `CANN.jl`                       | The neural network implementation   |
| `rap1_utils.jl`                 | Utility functions specific to RAP1 transcription factor site binding (as a test case) |
| `rap1-binder.jl`                | The driver code specific to this RAP1 binding |
| `rap1-lieb-positives.txt`       | Positive training examples   |
| `rap1-lieb-test.txt`            | Final holdout sequences for validation |
| `yeast-upstream-1k-negative.fa` | Sequences used to generate negative training examples |
| `README.md`                     | This file |

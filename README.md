# juliaNN

This is a neural network package for Julia implemented so I could better understand both Julia and artificial neural networks.

### Files
1. autoincoder.jl -- Generates a 8x3x8 autoencoder 
2. CANN_utils.jl -- Some neural network utility functions (read/write/encode/decode, etc..)  
3. CANN.jl -- The neural network implementation  
4. rap1_utils.jl -- Utility functions specific to RAP1 transcription factor site binding (as a test case)
5. rap1-binder.jl -- The driver code specific to this RAP1 binding
6. rap1-lieb-positives.txt -- Positive training examples  
7. rap1-lieb-test.txt -- Final holdout sequences for validation  
8. yeast-upstream-1k-negative.fa -- Sequences used to generate negative training examples
9. README.md -- This file

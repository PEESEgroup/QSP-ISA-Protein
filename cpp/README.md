# ISA, AA-VQC, and ADAPT-VQE Implementations

## Iterated Sparse Approximation Files
- **`isa_main.cpp/hpp`**: This is the main script for running the ISA algorithm. 
- **`simulator.cpp/hpp`**: This file contains functions for simulating the application of quantum gates to quantum states.
- **`state_tracker.cpp/hpp`**: This file is a wrapper around **`simulator.cpp/hpp`** and implements the iterative merging step for ISA. 
- **`sp3.cpp/hpp`**: This file implements 3-qubit state preparation, which is the base case for 2-bit patterns (k=2).
- **`greedy.cpp/hpp`**: This file implements ISA for k=1.
- **`greedy3.cpp/hpp`**: This file implements ISA for k=2.
- **`pattern.cpp/hpp`**: This file implements the pattern data structure.

## Alternating Ansatz - Variational Quantum Circuit Files
- **`vqc.cpp/hpp`**: This file implements gradient descent for AA-VQC based on this paper: https://arxiv.org/abs/2011.06258. 
- **`vqc_main.cpp/hpp`**: This is the main script for running the AA-VQC algorithm. 

## Adaptive Derivative-Assembled Pseudo-Trotter ansatz Variational Quantum Ei gensolver Files
- **`search.cpp/hpp`**: This file implements ADAPT-VQE based on this paper: https://www.nature.com/articles/s41467-019-10988-2. 
- **`search_main.cpp/hpp`**: This is the main script for running the ADAPT-VQE algorithm.










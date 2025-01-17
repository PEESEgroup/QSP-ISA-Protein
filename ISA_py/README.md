# Python Implementation of ISA

- **`qc.py`**: This file defines the Gate class and how rotations and CX gates are applied to quantum states. Gates are validated by qiskit's statevector simulator.
- **`sp3_extension`**: This file implements 3-qubit state preparation, which is the base case for 2-bit patterns (k=2).
- **`parser.py`**: This file defines function to extract state from properly formatted text file. 
- **`util.py`**: This file contains more helpers.
- **`isa.py`**: This file implements the enumeration of patterns and iterative merging step. It combines files in folder to define isa.prepare() which outputs same gate sequence as C++ executable.
- **`py_demo.py`** This script runs isa.prepare() and outputs the ISA gate sequence.




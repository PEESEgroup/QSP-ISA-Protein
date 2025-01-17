# Protein Experiments

Scripts for executing QSP implementations located in the `/cpp` directory and running protein experiments.

## Files
- **`isa_cpp_demo.py`**, **`search_cpp_demo.py`**, **`vqc_cpp_demo.py`**  
  These files are the scripts for the following QSP methods:  
  - **ISA (Iterated Sparse Approximation)**: `isa_cpp_demo.py`  
  - **ADAPT-VQE (Adaptive Derivative-Assembled Pseudo-Trotter ansatz Variational Quantum Eigensolver)**: `search_cpp_demo.py`  
  - **AA-VQC (Altenrnating Ansatz - Variational Quantum Circuit)**: `vqc_cpp_demo.py`  
    Each script calls corresponding build files which compiles the corresponding C++ implementation from `/cpp`. 

  **Inputs**:  
  - Protein-encoded quantum states from `data_and_logs/Protein_states_100`.  
  - Fidelity threshold (>0.95)

  **Outputs**:  
  The QSP implementation generates a gate sequence, used to calculate and save:  
  - **CX Count**  
  - **Gate Count**
  - **Classical Runtime**  
  - **Fidelity**

  **Results**:  
  - Metrics are stored in `data_and_logs/Protein_Results`

- **`uniprot.py`**: This sscipt generates protein-encoded states as `.txt` files using Uniprot Prot5 embeddings. 
- **`ucg_exp.py`**: Conducts runtime experiments for **UCG (Uniformly Controlled Gates)**.



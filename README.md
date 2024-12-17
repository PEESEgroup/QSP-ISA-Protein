# README 
> For internal use, not final -Rod



All folders, except ISA+QSP_Protein, is Ralph's code prior to my collaboration on the project. I will leave it to Ralph to contribute to the README for his code, if requested by QST.

All protein and noisy simulation experiments along with the implementations for ISA and benchmak QSP methods are compiled in ISA+QSP_Protein

## ISA+QSP_Protein

  Ralph's contribution:
  - cpp - folder containint ISA, AA-VQC, and ADAPT-VQE
  - build_ - build files for python wrappers to cpp
  - isa_cpp_demo.py, vqc_cpp_demo.py, search_cpp_demo.py - originally python wrappers to cpp, later used for protein experiments
  - isa.py, qc.py, util.py, parser.py - python implementation of ISA. Not used for final experimentation

  Rod's contribution:
  - isa_cpp_demo.py, vqc_cpp_demo.py, search_cpp_demo.py - Protein experiments for ISA, AA-VQC, and ADAPT-VQE -> results compiled in Protein_Results
  - noisy_simulation.py - Noisy simulation experiments on 5_qubit_state.txt -> Plots
  - ucg.exp - UCG runtime experiments
  - uniprot.py - encoding proteins from Uniprot into quantum states -> Protein_states_100
### Manuscript
  - QSP_paper/IOP-QSP-protein - up-to-date manuscript source files
  






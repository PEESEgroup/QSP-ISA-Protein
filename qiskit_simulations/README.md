# qiskit_simulations

Files for ISA-based simulations on IBM devices.

## Files
- **`noisy_simulation.py`**: This script prepares 100 random 5-qubit states using ISA. Fidelity thresholds are varied, and ideal fidelity, noisy fidelity, and CX counts are calculated per state. The noisy simulations use IBM FakeMelbourneV2. The results are compiled in `data_and_logs/Plots`.
- **`5_qubit_state.txt`**: This file stores randomly generated states to be inputted into ISA.

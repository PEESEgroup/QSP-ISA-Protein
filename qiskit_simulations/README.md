# qiskit_simulations

Files for ISA-based simulations on IBM devices.

## Files
- **`noisy_simulation.py`**: Prepares 100 random 5-qubit states using ISA. Fidelity thresholds are varied, and calculating ideal fidelity, noisy fidelity, and CX counts are calculated per state. Noisy simulations use IBM FakeMelbourneV2. Results are in `data_and_logs/Plots`.
- **`5_qubit_state.txt`**: Stores randomly generated states for ISA input.

#include "simulator.hpp"
#include "graph.hpp"

gate_sequence greedy3_prepare(const quantum_state& state, int n_qubits, double fidelity, const Graph& g);

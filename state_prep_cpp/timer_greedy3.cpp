#include "greedy3.hpp"
#include "simulator.hpp"
#include <vector>
#include <iostream>

int main() {
    int n_qubits = 14;
    int count = 100;
    double fidelity = 0.95;
    std::vector<int> counts;
    counts.reserve(count);
    for(int i = 0; i < count; i++) {
        quantum_state state = random_state(n_qubits);
        gate_sequence gates = greedy3_prepare(state, n_qubits, fidelity);
        counts.push_back(cx_count(gates));
    }
    int total = 0;
    for(int c : counts) total += c;
    std::cout << total / count << std::endl;
}

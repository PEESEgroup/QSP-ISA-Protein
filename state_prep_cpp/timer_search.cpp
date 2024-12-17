
#include "search.hpp"
#include <vector>
#include <iostream>

int main() {
    int n_qubits = 8;
    int count = 100;
    double fidelity = 0.95;
    std::vector<int> totals;
    for(int i = 0; i < count; i++) {
        std::cout << i << std::endl;
        quantum_state state = random_state(n_qubits);
        std::vector<gate_sequence> gates = prepare_search(state, n_qubits, (1 << n_qubits), fidelity);
        totals.push_back(gates.size() - 1);
    }
    int total = 0;
    for(int t : totals) total += t;
    std::cout << total / count << std::endl;
}

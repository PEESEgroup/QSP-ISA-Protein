
#include "greedy3.hpp"
#include "simulator.hpp"

#include <iostream>

int main() {
    std::vector<int> n_qubits_list {5};
    double fidelity = 0.95;
    int count = 100;
    for(int n_qubits : n_qubits_list) {
        std::vector<int> totals;
        for(int i = 0; i < count; i++) {
            quantum_state state = random_state(n_qubits);
            gate_sequence gates = greedy3_prepare(state, n_qubits, fidelity);
            totals.push_back(cx_count(gates));
        }
        int min = totals[0];
        int max = 0;
        for(int c : totals) {
            if(c < min) min = c;
            if(c > max) max = c;
        }
        double mean = 0;
        for(int c : totals) mean += c;
        mean /= count;
    
        double stdev = 0;
        for(int c : totals) stdev += (c - mean) * (c - mean);
        stdev = sqrt(stdev / count);
        
        std::cout << "n qubits = " << n_qubits << std::endl;
        std::cout << "  mean: " << mean << std::endl;
        std::cout << "  stdev: " << stdev << std::endl;
        std::cout << "  min: " << min << std::endl;
        std::cout << "  max: " << max << std::endl;
    }
}


#include "search.hpp"
#include "simulator.hpp"

#include <iostream>
#include <vector>

int main() {
    std::vector<int> n_qubits_list {5, 6};
    double fidelity = 0.95;
    int count = 100;
    for(int n_qubits : n_qubits_list) {
        std::vector<int> totals;
        for(int i = 0; i < count; i++) {
            std::cout << "n = " << n_qubits << ", i = " << i << std::endl;
            quantum_state state = random_state(n_qubits);
            std::vector<gate_sequence> gates = prepare_search(state, n_qubits, 
              (1 << n_qubits), fidelity);
            totals.push_back(gates.size() - 1);
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

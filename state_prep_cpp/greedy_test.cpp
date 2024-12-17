
#include "simulator.hpp"
#include "greedy.hpp"
#include "vqc.hpp"
#include <iostream>
#include <array>
#include <vector>
#include <cmath>

int main() {
    std::array<int, 4> n_qubits = {3, 4, 5, 6};
    for(int n : n_qubits) {
        std::vector<int> totals;
        for(int i = 0; i < 100; i++) {
            quantum_state target = random_state(n);
            gate_sequence gates = prepare_greedy(target, n);
            totals.push_back(cx_count(gates));
        }
        double mean = 0;
        for(int c : totals) mean += c;
        mean /= 100;
        
        double stdev = 0;
        for(int c : totals) stdev += (c - mean) * (c - mean);
        stdev = sqrt(stdev / 100);

        int min = 10000;
        int max = 0;
        for(int c : totals) {
            if(c < min) min = c;
            if(c > max) max = c;
        }
        std::cout << "n qubits = " << n << std::endl;
        std::cout << "  mean: " << mean << std::endl;
        std::cout << "  stdev: " << stdev << std::endl;
        std::cout << "  min: " << min << std::endl;
        std::cout << "  max: " << max << std::endl;
    }
    return 0;
}


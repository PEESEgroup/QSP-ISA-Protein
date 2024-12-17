#include "vqc.hpp"
#include "simulator.hpp"

#include <vector>
#include <iostream>

int main() {
    int count = 100;
    int n_qubits = 10;
    double fidelity = 0.95;
    int min_cx = (1 << (n_qubits - 1));
    gate_sequence init;
    std::vector<double> angles;
    angles.reserve(count);
    for(int k = 0; k < n_qubits; k++) {
        init.push_back(Gate::RY(k, 0));
        init.push_back(Gate::RZ(k, 0));
    }
    for(int l = 0; cx_count(init) < min_cx; l++) {
        for(int k = (l % 2); k < n_qubits - 1; k++) {
            init.push_back(Gate::CX(k, k + 1));
            init.push_back(Gate::RY(k, 0));
            init.push_back(Gate::RZ(k, 0));
            init.push_back(Gate::RY(k + 1, 0));
            init.push_back(Gate::RZ(k + 1, 0));
        }
    }
    for(int i = 0; i < count; i++) {
        quantum_state state = random_state(n_qubits);
        gate_sequence trained = init;
        while(fitness(state, trained) < fidelity) {
            trained = gradient_descent(trained, state, 0.1);
        }
        angles.push_back(trained[0].angle);
    }
    double total = 0;
    for(double angle : angles) total += angle;
    std::cout << total << std::endl;
}

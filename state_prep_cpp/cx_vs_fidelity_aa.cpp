#include "simulator.hpp"
#include "vqc.hpp"
#include <iostream>
#include <vector>

int main() {
    int n_qubits = 6;
    int min_layers = 11;
    int max_layers = 13;
    int training_iterations = 1000;
    double step_size = 0.1;
    int count = 100;
    std::vector<gate_sequence> templates;
    for(int l = min_layers; l <= max_layers; l++) {
        gate_sequence gates;
        for(int i = 0; i < n_qubits; i++) {
            gates.push_back(Gate::RY(i, 0));
            gates.push_back(Gate::RZ(i, 0));
        }
        for(int j = 0; j < l; j++) {
            for(int i = j % 2; i < n_qubits - 1; i += 2) {
                gates.push_back(Gate::CX(i, i + 1));
                gates.push_back(Gate::RY(i, 0));
                gates.push_back(Gate::RZ(i, 0));
                gates.push_back(Gate::RY(i + 1, 0));
                gates.push_back(Gate::RZ(i + 1, 0));
            }
        }
        templates.push_back(gates);
    }
    std::vector<double> fidelities(templates.size(), 0);
    for(int i = 0; i < count; i++) {
        std::cout << i << std::endl;
        quantum_state state = random_state(n_qubits);
        for(int k = 0; k < templates.size(); k++) {
            gate_sequence gates = templates[k];
            for(int t = 0; t < training_iterations; t++) {
                gates = gradient_descent(gates, state, step_size);
            }
            fidelities[k] += fitness(state, gates) / count;
        }
    }
    std::cout << "layers    cx count    fidelity" << std::endl;
    for(int k = 0; k < templates.size(); k++) {
        std::cout << k + min_layers << "         " << cx_count(templates[k])
            << "          " << fidelities[k] << std::endl;
    }
    return 0;
}


#include "genetic.hpp"
#include <complex>
#include <algorithm>
#include <iostream>

double fitness(quantum_state, gate_sequence);
gate_sequence apply_genetic(quantum_state, gate_sequence, double, int, int, int);

int main() {
    int n_qubits = 6;
    std::vector<Gate> seq;
    for(int i = 0; i < 6; i++) {
        for(int i = 0; i < n_qubits; i++) {
            seq.push_back(Gate::RY(i, 0));
            seq.push_back(Gate::RZ(i, 0));
        }
        for(int i = 0; i < n_qubits - 1; i++) {
            seq.push_back(Gate::CX(i, i + 1));
            seq.push_back(Gate::RY(i, 0));
            seq.push_back(Gate::RZ(i, 0));
            seq.push_back(Gate::RY(i + 1, 0));
            seq.push_back(Gate::RX(i + 1, 0));
        }
    }
    quantum_state target_state = random_state(6);
    std::cout << "start fitness: " << fitness(target_state, seq) << std::endl;
    seq = apply_genetic(target_state, seq, 0.2, 3, 3, 500);
    std::cout << "final fitness: " << fitness(target_state, seq) << std::endl;
}

gate_sequence mutate_sequence(gate_sequence seq, double step_size) {
    gate_sequence output;
    for(int i = 0; i < seq.size(); i++) {
        output.push_back(seq[i].mutate(step_size));
    }
    return output;
}

double fitness(quantum_state start, gate_sequence seq) {
    quantum_state state = start;
    for(const Gate& g : seq) {
        state = apply_gate(g, state);
    }
    return abs(state[0]);
}

gate_sequence apply_genetic(quantum_state start, gate_sequence seq,
  double step_size, int num_survivors, int num_offspring, int num_iterations) {
    std::vector<gate_sequence> output = {seq};
    for(int i = 0; i < num_iterations; i++) {
        int num_parents = output.size();
        for(int j = 0; j < num_parents; j++) {
            for(int k = 0; k < num_offspring; k++) {
                output.push_back(mutate_sequence(output[j], step_size));
            }
        }
        if(output.size() <= num_survivors) continue;
        std::vector<double> values = std::vector<double>(output.size());
        for(int i = 0; i < output.size(); i++) {
            values[i] = fitness(start, output[i]);
        }
        std::vector<int> indices = std::vector<int>(output.size());
        for(int i = 0; i < output.size(); i++) {
            indices[i] = i;
        }
        std::sort(indices.begin(), indices.end(), 
          [&values](int i, int j) {return values[j] < values[i];});
        std::vector<gate_sequence> survivors(num_survivors);
        for(int i = 0; i < num_survivors; i++) {
            survivors[i] = output[indices[i]];
        }
        output = survivors;
        std::cout << fitness(start, output[0]) << std::endl;
    }
    return output[0];
}

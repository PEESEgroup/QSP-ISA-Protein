#include "simulator.hpp"
#include "vqc.hpp"
#include <iostream>
#include <vector>

#include <mutex>
#include <thread>

std::mutex state_lock;
std::mutex counts_lock;

void progress_bar(int completed, int total) {
    auto now = std::chrono::system_clock::now();
    std::time_t nowtime = std::chrono::system_clock::to_time_t(now);
    std::cout << std::ctime(&nowtime) << ": Completed "
      << completed << "/" << total << std::endl;
}

void compute_fidelities(const std::vector<gate_sequence>& templates, 
  int n_qubits, int counts, int state_buffer_size, int training_iterations,
  double step_size, std::vector<double>& fidelities, bool print_progress) {
    std::vector<double> output(templates.size(), 0);
    std::vector<quantum_state> state_buffer;
    int index = 1;
    for(int i = 0; i < counts; i++) {
        if(index >= state_buffer.size()) {
            index = 0;
            state_buffer.clear();
            state_lock.lock();
            for(int j = 0; j < state_buffer_size; j++) {
                state_buffer.push_back(random_state(n_qubits));
            }
            state_lock.unlock();
        }
        quantum_state state = state_buffer[index];
        for(int j = 0; j < templates.size(); j++) {
            gate_sequence gates = templates[j];
            for(int k = 0; k < training_iterations; k++) {
                gates = gradient_descent(gates, state, step_size);
            }
            output[j] += fitness(state, gates);
        }
        if(print_progress) progress_bar(i, counts);
    }
    counts_lock.lock();
    for(int i = 0; i < output.size(); i++) {
        fidelities[i] += output[i];
    }
    counts_lock.unlock();
}

int num_training_iterations(int n_qubits) {
    quantum_state state = random_state(n_qubits);
    gate_sequence gates;
    for(int i = 0; i < n_qubits; i++) {
        gates.push_back(Gate::RY(i, 0));
        gates.push_back(Gate::RZ(i, 0));
    }
    for(int j = 0; cx_count(gates) < (1 << (n_qubits - 1)); j++) {
        for(int i = j & 1; i < n_qubits - 1; i += 2) {
            gates.push_back(Gate::CX(i, i + 1));
            gates.push_back(Gate::RY(i, 0));
            gates.push_back(Gate::RZ(i, 0));
            gates.push_back(Gate::RY(i + 1, 0));
            gates.push_back(Gate::RZ(i + 1, 0));
        }
    }
    std::cout << "CX gate count: " << cx_count(gates) << std::endl;
    int output = 0;
    while(fitness(state, gates) < 0.95) {
        gates = gradient_descent(gates, state, 0.1);
        output++;
    }
    return 2 * output;
}

int main() {
    int n_qubits = 8;
    int min_layers = 29;
    int max_layers = 31;
    int training_iterations = num_training_iterations(n_qubits)
      + num_training_iterations(n_qubits) + num_training_iterations(n_qubits);
    std::cout << "Number of training iterations used: " << training_iterations << std::endl;
    double step_size = 0.1;
    int count = 100;
    int num_threads = 4;
    int state_buffer_size = 5;
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
    int states_per_thread = count / num_threads;
    int remainder = count % num_threads;
    std::vector<std::thread> threads;
    for(int t = 0; t < num_threads; t++) {
        int instance_count = states_per_thread;
        if(t == 0) instance_count += remainder;
        threads.emplace_back(compute_fidelities, std::ref(templates), n_qubits,
          instance_count, state_buffer_size, training_iterations, step_size, 
          std::ref(fidelities), t == 0);
    }
    for(int t = 0; t < num_threads; t++) {
        threads[t].join();
    }
    std::cout << "layers    cx count    fidelity" << std::endl;
    for(int k = 0; k < templates.size(); k++) {
        std::cout << k + min_layers << "         " 
          << cx_count(templates[k]) << "          "
          << fidelities[k] / count << std::endl;
    }
}

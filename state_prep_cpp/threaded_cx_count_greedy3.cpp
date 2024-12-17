#include "greedy3.hpp"
#include "simulator.hpp"

#include <iostream>
#include <thread>
#include <mutex>
#include <vector>
#include <ctime>
#include <chrono>

std::mutex state_lock;
std::mutex counts_lock;

void progress_bar(int completed, int total) {
    auto now = std::chrono::system_clock::now();
    std::time_t nowtime = std::chrono::system_clock::to_time_t(now);
    std::cout << std::ctime(&nowtime) << ": Completed "
      << completed << "/" << total << std::endl;
}

void compute_counts(int n_qubits, double fidelity, int count,
  int state_buffer_size, std::vector<int>& counts, bool print_progress) {
    std::vector<int> output;
    std::vector<quantum_state> state_buffer;
    int index = 1;
    while(output.size() < count) {
        if(index >= state_buffer.size()) {
            index = 0;
            state_buffer.clear();
            state_lock.lock();
            for(int i = 0; i < state_buffer_size; i++) {
                state_buffer.push_back(random_state(n_qubits));
            }
            state_lock.unlock();
        }
        gate_sequence gates = greedy3_prepare(state_buffer[index], n_qubits, 
          fidelity);
        output.push_back(cx_count(gates));
        index++;
        if(print_progress) progress_bar(output.size(), count);
    }
    counts_lock.lock();
    for(int c : output) counts.push_back(c);
    counts_lock.unlock();
}

int main() {
    int n_qubits = 16;
    double fidelity = 0.95;
    int count = 100;
    int num_threads = 8;
    int states_per_thread = count / num_threads;
    int remainder = count % num_threads;
    int buffer_size = 4;
    std::vector<int> counts;
    std::vector<std::thread> threads;
    for(int t = 0; t < num_threads; t++) {
        int instance_count = states_per_thread;
        if(t == 0) instance_count += remainder;
        threads.emplace_back(compute_counts, n_qubits, fidelity, 
          instance_count, buffer_size, std::ref(counts), t == 0);
    }
    for(int t = 0; t < num_threads; t++) {
        threads[t].join();
    }
    int min = counts[0];
    int max = 0;
    for(int c : counts) {
        if(c < min) min = c;
        if(c > max) max = c;
    }
    double mean = 0;
    for(int c : counts) mean += c;
    mean /= count;

    double stdev = 0;
    for(int c : counts) stdev += (c - mean) * (c - mean);
    stdev = sqrt(stdev / count);

    std::cout << "Count: " << counts.size() << std::endl;
    std::cout << "  mean: " << mean << std::endl;
    std::cout << "  stdev: " << stdev << std::endl;
    std::cout << "  min: " << min << std::endl;
    std::cout << "  max: " << max << std::endl;
}

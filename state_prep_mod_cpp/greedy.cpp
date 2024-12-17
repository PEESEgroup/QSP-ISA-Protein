
#include "greedy.hpp"
#include "simulator.hpp"
#include "state_tracker.hpp"
#include <algorithm>
#include <vector>
#include <iostream>
#include <complex>

int max_index(const std::vector<double>& list) {
    int output = 0;
    double value = list[0];
    for(int i = 1; i < list.size(); i++) {
        if(list[i] > value) {
            value = list[i];
            output = i;
        }
    }
    return output;
}

void greedy_phase1(StateTracker& tracker, int n_qubits) {
    std::vector<int> candy1;
    for(int i = 0; i < n_qubits; i++) candy1.push_back(i);
    std::vector<double> amps = magnitudes(tracker.state);
    int index = max_index(amps);
    while(candy1.size() > 0) {
        std::vector<double> values;
        for(int c : candy1) {
            values.push_back(abs(tracker.state[index ^ (1 << c)]));
        }
        int target = candy1[max_index(values)];
        int src = std::max(index, index ^ (1 << target));
        int dest = std::min(index, index ^ (1 << target));
        tracker.rotate_merge(src, dest);
        candy1.erase(std::remove(candy1.begin(), candy1.end(), target), 
            candy1.end());
        index = dest;
    }
}


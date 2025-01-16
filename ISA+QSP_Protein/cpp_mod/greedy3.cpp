
#include "greedy3.hpp"
#include "sp3.hpp"
#include "state_tracker.hpp"
#include "pattern.hpp"
#include "simulator.hpp"
#include "greedy.hpp"
#include <vector>
#include <iostream>
#include <algorithm>

gate_sequence greedy3_prepare(const quantum_state& state, int n_qubits, 
  double target_fidelity, const Graph& graph) {
    StateTracker tracker(state);
    greedy_phase1(tracker, n_qubits);
    std::map<Pattern, int> patterns = list_patterns(graph);
    uint32_t last2 = 0;
    while(abs(tracker.state[0]) * abs(tracker.state[0]) < target_fidelity) {
        Pattern p_selected(0, 0, 1);
        double max_value = -1;
        for(const auto& pair : patterns) {
            Pattern p = pair.first;
            double discount = 0;
            if(p.bits == 0) {
                discount = abs(tracker.state[0]) * abs(tracker.state[0]);
            }
            double subs_norm = norm(substate(tracker.state, p));
            double increase = subs_norm * subs_norm - discount;
            double value = increase / (pair.second + 1);
            if(value > max_value) {
                max_value = value;
                p_selected = p;
            }
        }
        if(patterns.at(p_selected) == 2 && last2 == p_selected.wilds) {
            tracker.undo_block();
        }
        tracker.new_block();
        while(patterns.at(p_selected) > 3) {
            std::vector<std::pair<int, int> > targets = list_cx(p_selected, graph);
            std::vector<double> t_values;
            quantum_state subs1 = substate(tracker.state, p_selected);
            double subs1_norm = norm(subs1);
            double subs1_mag = subs1_norm * subs1_norm;
            for(std::pair<int, int> t : targets) {
                Pattern other = apply_cx(p_selected, t.first, t.second);
                quantum_state subs2 = substate(tracker.state, other);
                double subs2_norm = norm(subs2);
                double subs2_mag = subs2_norm * subs2_norm;
                std::complex<double> cross = 0;
                for(int i = 0; i < subs1.size(); i++) {
                    cross += conj(subs1[i]) * subs2[i];
                }
                double subs12_mag = abs(cross);
                double A = 0.5 * (subs1_mag - subs2_mag);
                double value = 0.5 * (subs1_mag + subs2_mag) 
                    + sqrt(A * A + subs12_mag * subs12_mag);
                int cost = std::max(patterns.at(other), 
                  patterns.at(p_selected));
                t_values.push_back(value / cost);
            }
            std::pair<int, int> target = targets[max_index(t_values)];
            Pattern src = p_selected;
            Pattern dest = apply_cx(p_selected, target.first, target.second);
            if(patterns.at(src) < patterns.at(dest)) {
                src = dest;
                dest = p_selected;
            }
            p_selected = dest;
            tracker.control_rotate_merge_chunk(target.first, src, dest);
        }
        if(patterns.at(p_selected) == 3) {
            int q0, q1, q2;
            int bit = wild_indices(p_selected.bits)[0];
            std::vector<int> wild_bits = wild_indices(p_selected.wilds);
            if(graph.has_edge(bit, wild_bits[0])) {
                q0 = wild_bits[1];
                q1 = wild_bits[0];
                q2 = bit;
            } else {
                q0 = wild_bits[0];
                q1 = wild_bits[1];
                q2 = bit;
            }
            last2 = p_selected.wilds;
            tracker.new_block();
            sp3_1(tracker, q0, q1, q2);
            tracker.new_block();
            sp2(tracker, q0, q1);
        } else if(patterns.at(p_selected) == 1) {
            std::vector<int> wild_bits = wild_indices(p_selected.wilds);
            int q0 = wild_bits[0];
            int q1 = wild_bits[1];
            sp2(tracker, q0, q1);
            last2 = p_selected.wilds;
        } else {
            int target = wild_indices(p_selected.wilds)[0];
            tracker.rotate_merge((1 << target), 0);
            last2 = 0;
        }
    }
    return tracker.gates_seq();
}


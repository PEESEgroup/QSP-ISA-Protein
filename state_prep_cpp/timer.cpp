#include "greedy.hpp"
#include "simulator.hpp"
#include "vqc.hpp"
#include <vector>
#include <iostream>
#include <algorithm>

gate_sequence test_greedy(const quantum_state&, double, int);
//gate_sequence test_greedy_vqc(const quantum_state&, double, int);
gate_sequence test_vqc(const quantum_state&, double, int);
gate_sequence test_vqc_search(const quantum_state&, double, int);

int main() {
    int n_qubits = 6;
    int count = 100;
    double fidelity = 0.95;
    std::vector<double> cx_counts;
    cx_counts.reserve(count);
    for(int i = 0; i < count; i++) {
        quantum_state target = random_state(n_qubits);
        gate_sequence seq = test_vqc(target, fidelity, n_qubits);
        cx_counts.push_back(cx_count(seq));
    }
    for(const double& d : cx_counts) {
        std::cout << d << "\n";
    }
}

gate_sequence test_greedy(const quantum_state& target, double fidelity, 
    int n_qubits) {
    StateTracker tracker(target);
    greedy_phase1(tracker, n_qubits);
    while(abs(tracker.state[0]) * abs(tracker.state[0]) < fidelity) {
        greedy_phase2_iter(tracker, n_qubits);
    }
    return tracker.gates_seq();
}

gate_sequence test_vqc(const quantum_state& target, double fidelity, 
    int n_qubits) {
    gate_sequence output;
    for(int i = 0; i < n_qubits; i++) {
        output.push_back(Gate::RY(i, 0));
        output.push_back(Gate::RZ(i, 0));
    }
    for(int l = 0; l < 10; l++) {
        for(int i = l % 2; i < n_qubits - 1; i += 2) {
            output.push_back(Gate::CX(i, i + 1));
            output.push_back(Gate::RY(i, 0));
            output.push_back(Gate::RZ(i, 0));
            output.push_back(Gate::RY(i + 1, 0));
            output.push_back(Gate::RX(i + 1, 0));
        }
    }
    while(fitness(target, output) < fidelity) {
        output = gradient_descent(output, target, 0.1);
    }
    return output;
}

gate_sequence test_vqc_search(const quantum_state& target, double fidelity,
    int n_qubits) {
    std::vector<gate_sequence> candy;
    gate_sequence seq;
    for(int i = 0; i < n_qubits; i++) {
        seq.push_back(Gate::RY(i, 0));
        seq.push_back(Gate::RZ(i, 0));
    }
    for(int i = 0; i < 100; i++) {
        seq = gradient_descent(seq, target, 0.1);
    }
    candy.push_back(seq);
    while(fitness(target, candy[0]) < fidelity) {
        int parents = candy.size();
        for(int i = 0; i < parents; i++) {
            for(int j = 0; j < n_qubits - 1; j++) {
                gate_sequence child;
                child.push_back(Gate::RZ(j, 0));
                child.push_back(Gate::RY(j, 0));
                child.push_back(Gate::RX(j + 1, 0));
                child.push_back(Gate::RY(j + 1, 0));
                child.push_back(Gate::CX(j, j + 1));
                child.insert(child.end(), candy[i].begin(), candy[i].end());
                for(int i = 0; i < 100; i++) {
                    child = gradient_descent(child, target, 0.1);
                }
                candy.push_back(child);
            }
        }
        std::vector<double> fids;
        for(const gate_sequence& g : candy) {
            fids.push_back(fitness(target, g));
        }
        std::vector<double> indices;
        for(int i = 0; i < candy.size(); i++) {
            indices.push_back(i);
        }
        std::sort(indices.begin(), indices.end(), [&fids](int i, int j) {
            return fids[j] < fids[i];
        });
        std::vector<gate_sequence> temp;
        for(int i = 0; i < 3 && i < candy.size(); i++) {
            temp.push_back(candy[indices[i]]);
        }
        candy = temp;
    }
    return candy[0];
}

//gate_sequence test_greedy_vqc(const quantum_state& target, double fidelity,
//    int n_qubits) {
//    StateTracker tracker(target);
//    greedy_phase1(tracker, n_qubits);
//    while(true) {
//        gate_sequence gates = tracker.prep_sequence();
//        double current_fidelity = fitness(target, gates);
//        for(int i = 0; i < 200; i++) {
//            gate_sequence ng = gradient_descent(gates, target, 0.1);
//            double nf = fitness(target, ng);
//            if(nf < current_fidelity) break;
//            current_fidelity = nf;
//            gates = ng;
//        }
//        gate_sequence tg;
//        tg.reserve(gates.size());
//        for(int i = gates.size() - 1; i >= 0; i--) {
//            tg.push_back(gates[i].inverse());
//        }
//        tracker = StateTracker(target, tg);
//        if(tracker.amps[0] * tracker.amps[0] > fidelity) break;
//        greedy_phase2_iter(tracker, n_qubits);
//    }
//    return tracker.prep_sequence();
//}

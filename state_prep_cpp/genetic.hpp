
#pragma once
#include "simulator.hpp"
#include <vector>

using gate_sequence = std::vector<Gate>;

gate_sequence apply_genetic(quantum_state start, gate_sequence seq, 
    double step_size, int num_survivors, int num_offspring, int num_iterations);


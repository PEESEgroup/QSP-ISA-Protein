
#include "simulator.hpp"
#include "vqc.hpp"
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <optional>

/**
 * This file interprets the commands in a specified file according to the
 * following specification:
 * 
 *   n_qubits: sets the number of qubits for the script
 *
 *   init_state: interprets the following lines as the specification of a
 *   quantum state, until [end] command is reached. Then sets the state of
 *   the quantum computer to the specified state. Each spec line is formatted as
 *   [number] [amplitude] [phase]. [number] should be a binary bitstring.
 *   
 *   end: used to denote the end of the init_state command
 *   rx, ry, rz, cx: quantum gates
 *   print: prints everything in this line
 *   print_state: prints the current quantum state
 *   print_gradient: computes and prints the gradient of the gate sequence up
 *     to this point, assuming the target state is |0>
 *   //: comment line
 * 
 * Usage: ./a.out [file]
 */

std::vector<std::string> tokenize(const std::string& s) {
    std::vector<std::string> output;
    std::vector<char> buffer;
    for(int i = 0; i < s.length(); i++) {
        if(isspace(s[i])) {
            if(buffer.size() > 0) {
                output.push_back(std::string(buffer.begin(), buffer.end()));
                buffer.clear();
            }
        } else {
            buffer.push_back(s[i]);
        }
    }
    if(buffer.size() > 0) {
        output.push_back(std::string(buffer.begin(), buffer.end()));
    }
    return output;
}

int bin_string_to_int(const std::string& str) {
    int output = 0;
    for(int i = 0; i < str.length(); i++) {
        if(str[i] == '1') output += (1 << (str.length() - i - 1));
    }
    return output;
}

class InterpreterState {
    private:
        int n_qubits;
        quantum_state start;
        quantum_state state;
        gate_sequence gates;
        bool initializing;
    public:
        InterpreterState() {
            this->n_qubits = -1;
            this->initializing = false;
        }
        void set_n_qubits(int n_qubits) {
            if(this->initializing) {
                throw std::runtime_error("missing end statement");
            }
            if(n_qubits <= 0) {
                throw std::runtime_error("n_qubits must be positive");
            }
            this->n_qubits = n_qubits;
            this->start = start_state(n_qubits);
            this->state = this->start;
        }
        void init_state() {
            if(this->initializing) {
                throw std::runtime_error("missing end statement");
            }
            if(n_qubits <= 0) {
                throw std::runtime_error("n_qubits not specified");
            }
            this->initializing = true;
            this->start = start_state(this->n_qubits);
            this->gates.clear();
        }
        void state_entry(const std::string& position, double amp, double phase) {
            if(!this->initializing) {
                throw std::runtime_error("missing init_state statement");
            }
            if(position.length() != this->n_qubits) {
                throw std::runtime_error("wrong number of qubits");
            }
            int pos = bin_string_to_int(position);
            this->start[pos] = std::complex<double>(amp * cos(phase), amp * sin(phase));
        }
        void end() {
            if(!this->initializing) {
                throw std::runtime_error("unexpected end statement");
            }
            if(abs(norm(this->start) - 1) > 0.0001) {
                throw std::runtime_error("non-normalized start state");
            }
            this->state = this->start;
            this->initializing = false;
        }
        inline void check_initialized() {
            if(this->initializing) {
                throw std::runtime_error("state entry or end expected");
            }
            if(this->n_qubits <= 0) {
                throw std::runtime_error("state not initialized");
            }
        }
        void print_state_() {
            this->check_initialized();
            print_state(this->state, this->n_qubits);
            std::cout << std::endl;
        }
        void apply_gate_(const Gate& g) {
            this->check_initialized();
            this->state = apply_gate(g, this->state);
            this->gates.push_back(g);
        }
        void print_gradient() {
            this->check_initialized();
            quantum_state target = start_state(this->n_qubits);
            std::vector<double> gradient = compute_gradient(this->gates, this->start, target);
            for(int i = 0; i < gradient.size(); i++) {
                std::cout << this->gates[i].to_string() << " " 
                    << gradient[i] << std::endl;
            }
            std::cout << std::endl;
        }
};

void syntax_error() {
    throw std::runtime_error("syntax error");
}

int main(int argc, char* argv[]) {
    if(argc != 2) {
        std::cout << "Usage: ./interpret [file]" << std::endl;
        return 0;
    }
    std::ifstream file(argv[1]);
    std::string line;
    int line_num = 0;
    InterpreterState ints;
    std::cout << "Reading file: " << argv[1] << std::endl;
    while(getline(file, line)) {
        line_num++;
        std::vector<std::string> tokens = tokenize(line);
        if(tokens.size() == 0) continue;
        std::string cmd = tokens[0];
        if(cmd.length() >= 2 and cmd[0] == '/' and cmd[1] == '/') continue;
        try {
            if(cmd == "n_qubits") {
                if(tokens.size() != 2) {
                    syntax_error();
                }
                int n = stoi(tokens[1]);
                ints.set_n_qubits(n);
                continue;
            }
            if(cmd == "init_state") {
                ints.init_state();
                continue;
            }
            if(cmd == "end") {
                ints.end();
                continue;
            }
            if(cmd == "rx") {
                if(tokens.size() != 3) syntax_error();
                int target = stoi(tokens[1]);
                double angle = stod(tokens[2]);
                ints.apply_gate_(Gate::RX(target, angle));
                continue;
            }
            if(cmd == "ry") {
                if(tokens.size() != 3) syntax_error();
                int target = stoi(tokens[1]);
                double angle = stod(tokens[2]);
                ints.apply_gate_(Gate::RY(target, angle));
                continue;
            }
            if(cmd == "rz") {
                if(tokens.size() != 3) syntax_error();
                int target = stoi(tokens[1]);
                double angle = stod(tokens[2]);
                ints.apply_gate_(Gate::RZ(target, angle));
                continue;
            }
            if(cmd == "cx") {
                if(tokens.size() != 3) syntax_error();
                int control = stoi(tokens[1]);
                int target = stoi(tokens[2]);
                ints.apply_gate_(Gate::CX(control, target));
                continue;
            }
            if(cmd == "print") {
                int i = line.find("t");
                std::cout << line.substr(i + 1) << std::endl;
                continue;
            }
            if(cmd == "print_state") {
                ints.print_state_();
                continue;
            }
            if(cmd == "print_gradient") {
                ints.print_gradient();
                continue;
            }
            if(tokens.size() != 3) syntax_error();
            ints.state_entry(tokens[0], stod(tokens[1]), stod(tokens[2]));
        } catch(const std::runtime_error& e) {
            throw std::runtime_error("Line " + std::to_string(line_num) + ": " 
                + e.what());
        }
    }
    return 0;
}

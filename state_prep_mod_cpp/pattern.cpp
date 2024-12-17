#include "pattern.hpp"
#include <queue>
#include <iostream>

//int main() {
//    Graph g(4);
//    g.add_edge(0, 1);
//    g.add_edge(1, 2);
//    g.add_edge(2, 3);
//    std::map<Pattern, int> patterns = list_patterns(g);
//    for(const auto& pair : patterns) {
//        std::cout << pair.first.to_string() << "    " << pair.second << std::endl;
//    }
//    Pattern p(1, 6, 4);
//    quantum_state state;
//    for(int i = 0; i < 16; i++) state.push_back(i);
//    print_state(state, 4);
//    quantum_state subs = substate(state, p);
//    std::cout << subs.size() << std::endl;
//    print_state(subs, 2);
//}

Pattern::Pattern(uint32_t bits, uint32_t wilds, int n):
  bits(bits & (~wilds)), wilds(wilds), n(n) {
    if(n < 1 || n > 32) {
        throw std::runtime_error("n too large");
    }
    if((this->bits & ((1 << n) - 1)) != this->bits) {
        throw std::runtime_error("Too many bits");
    }
    if((this->wilds & ((1 << n) - 1)) != this->wilds) {
        throw std::runtime_error("Too many wilds");
    }
    if((this->wilds & this->bits) != 0) {
        throw std::runtime_error("Class invariant violated");
    }
}

Pattern Pattern::from_vector(std::vector<BitType> types) {
    uint32_t bits = 0;
    uint32_t wilds = 0;
    int n = types.size();
    for(int i = 0; i < n; i++) {
        if(types[i] == BitType::ONE) bits = bits ^ (1 << i);
        if(types[i] == BitType::WILD) wilds = wilds ^ (1 << i);
    }
    return Pattern(bits, wilds, n);
}

BitType Pattern::get_bit(int index) const {
    if((this->wilds >> index) & 1) return BitType::WILD;
    return (this->bits >> index) & 1 ? ONE : ZERO;
}

bool Pattern::operator==(const Pattern& other) const {
    return this->bits == other.bits
      && this->wilds == other.wilds
      && this->n == other.n;
}

bool Pattern::operator<(const Pattern& other) const {
    if(this->n < other.n) return true;
    if(this->n > other.n) return false;
    if(this->bits < other.bits) return true;
    if(this->bits > other.bits) return false;
    return this->wilds < other.wilds;
}

std::string Pattern::to_string() const {
    std::vector<char> chars;
    for(int i = 0; i < this->n; i++) {
        BitType bit = this->get_bit(i);
        if(bit == BitType::ONE) chars.push_back('1');
        if(bit == BitType::ZERO) chars.push_back('0');
        if(bit == BitType::WILD) chars.push_back('*');
    }
    return std::string(chars.begin(), chars.end());
}

Pattern apply_cx(const Pattern& pattern, int control, int target) {
    BitType control_bit = pattern.get_bit(control);
    BitType target_bit = pattern.get_bit(target);
    if(control_bit == BitType::WILD || target_bit == BitType::WILD) {
        throw std::runtime_error("Invalid CX operation");
    }
    if(control_bit == BitType::ZERO) return pattern;
    uint32_t new_bit = pattern.bits ^ (1 << target);
    return Pattern(new_bit, pattern.wilds, pattern.n);
}

std::vector<std::pair<int, int> > list_cx(const Pattern& p, const Graph& g) {
    std::vector<std::pair<int, int> > output(0);
    for(int control = 0; control < p.n; control++) {
        if(p.get_bit(control) != BitType::ONE) {
            continue;
        }
        for(const int target : g.list_edges(control)) {
            if(p.get_bit(target) != BitType::WILD) {
                output.push_back(std::pair<int, int> (control, target));
            }
        }
    }
    return output;
}

std::map<Pattern, int> list_patterns(const Graph& g) {
    //first add all the single * patterns
    std::map<Pattern, int> output;
    int n = g.n;
    for(int i = 0; i < n; i++) {
        Pattern p(0, (1 << i), n);
        output.emplace(p, 0);
    }
    //next, iterate through all connected pairs of qubits
    for(int i = 0; i < n; i++) {
        for(int j = i + 1; j < n; j++) {
            if(!g.has_edge(i, j)) continue;
            uint32_t wilds = (1 << i) | (1 << j);
            Pattern base(0, wilds, n);
            output.emplace(base, 1);
            //now set up the queue of base 3-qubit patterns
            std::queue<std::pair<Pattern, int> > q;
            for(const int k : g.list_edges(i)) {
                q.push(std::pair<Pattern, int>(Pattern(1 << k, wilds, n), 3));
            }
            for(const int k : g.list_edges(j)) {
                q.push(std::pair<Pattern, int>(Pattern(1 << k, wilds, n), 3));
            }
            //now do breadth first search
            while(!q.empty()) {
                std::pair<Pattern, int> next = q.front();
                q.pop();
                if(output.find(next.first) != output.end()) continue;
                output.insert(next);
                std::vector<std::pair<int, int> > cxs = list_cx(next.first, g);
                for(const std::pair<int, int>& cx : cxs) {
                    Pattern np = apply_cx(next.first, cx.first, cx.second);
                    q.push(std::pair<Pattern, int>(np, next.second + 1));
                }
            }
        }
    }
    return output;
}

std::vector<int> wild_indices(uint32_t wild) {
    std::vector<int> output;
    int i = 0;
    while(wild > 0) {
        if(wild & 1) output.push_back(i);
        wild = wild >> 1;
        i++;
    }
    return output;
}

quantum_state substate(const quantum_state& state, const Pattern& p) {
    quantum_state output;
    std::vector<int> indices = wild_indices(p.wilds);
    for(int i = 0; i < (1 << indices.size()); i++) {
        int index = p.bits;
        for(int j = 0; j < indices.size(); j++) {
            index += ((i >> j) & 1) * (1 << indices[j]);
        }
        output.push_back(state[index]);
    }
    return output;
}


#pragma once

#include "graph.hpp"
#include "simulator.hpp"
#include <vector>
#include <map>

enum BitType {
    ONE, ZERO, WILD
};

/**
 * Implementation note: only the lower [n] bits of [bits] and [wilds] are
 * allowed to be nonzero. Also, any bit positions where [wilds] is one, [bits]
 * is zero.
 */
class Pattern {
    public:
        Pattern(uint32_t bits, uint32_t wilds, int n);
        static Pattern from_vector(std::vector<BitType> types);
        BitType get_bit(int index) const;
        uint32_t bits;
        uint32_t wilds;
        int n;
        bool operator==(const Pattern& other) const;
        bool operator<(const Pattern& other) const;
        std::string to_string() const;
};

Pattern apply_cx(const Pattern& pattern, int control, int target);
std::vector<std::pair<int, int> > list_cx(const Pattern& p, const Graph& g);
std::map<Pattern, int> list_patterns(const Graph& g);
//returns a list of indices where [wild] has a one in its binary representation
std::vector<int> wild_indices(uint32_t wild);
quantum_state substate(const quantum_state& state, const Pattern& pattern);

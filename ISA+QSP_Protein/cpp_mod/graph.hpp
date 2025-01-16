
#pragma once
#include <vector>
#include <string>

class Graph {
    public:
        static Graph fromFile(std::string filename);
        Graph(int n);
        const int n;
        bool has_edge(int i, int j) const;
        std::vector<int> list_edges(int i) const;
        void add_edge(int i, int j);
        void remove_edge(int i, int j);
    private:
        std::vector<int> values;
};


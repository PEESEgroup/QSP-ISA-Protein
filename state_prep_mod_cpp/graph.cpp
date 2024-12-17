
#include "graph.hpp"
#include <vector>
#include <string>
#include <fstream>

bool checkBounds(int i, int j, int n) {
    return i >= 0 && i < n && j >= 0 && j < n;
}

Graph Graph::fromFile(std::string filename) {
    std::ifstream file(filename.c_str());
    int n;
    int m;
    std::string line;
    while(getline(file, line)) {
        if(line.length() == 0 || line.at(0) == '#') continue;
        if(sscanf(line.c_str(), "%d %d", &n, &m) != 2) {
            throw std::runtime_error("Illegal first line");
        }
        break;
    }
    Graph output(n);
    while(m > 0) {
        getline(file, line);
        if(line.length() == 0 || line.at(0) == '#') continue;
        int i, j;
        if(sscanf(line.c_str(), "%d %d", &i, &j) != 2) {
            throw std::runtime_error("Illegal line format");
        }
        if(output.has_edge(i, j)) {
            throw std::runtime_error("Duplicate initialization");
        }
        if(!checkBounds(i, j, n)) {
            throw std::runtime_error("Out of bounds");
        }
        output.add_edge(i, j);
        m--;
    }
    return output;
}

int compute_index(int i, int j, int n) {
    if(i <= j) return i * n - i * (i - 1) / 2 + j - i;
    return j * n - j * (j - 1) / 2 + i - j;
}

Graph::Graph(int n) : n(n) {
    this->values = std::vector<int>(n * (n - 1) / 2 + n, 0);
}

bool Graph::has_edge(int i, int j) const {
    if(!checkBounds(i, j, this->n)) {
        throw std::runtime_error("Out of bounds");
    }
    return this->values[compute_index(i, j, this->n)];
}

std::vector<int> Graph::list_edges(int i) const {
    std::vector<int> output;
    for(int k = 0; k < this->n; k++) {
        if(this->has_edge(i, k)) output.push_back(k);
    }
    return output;
}

void Graph::add_edge(int i, int j) {
    if(!checkBounds(i, j, this->n)) {
        throw std::runtime_error("Out of bounds");
    }
    this->values[compute_index(i, j, this->n)] = 1;
}

void Graph::remove_edge(int i, int j) {
    if(!checkBounds(i, j, this->n)) {
        throw std::runtime_error("Out of bounds");
    }
    this->values[compute_index(i, j, this->n)] = 0;
}


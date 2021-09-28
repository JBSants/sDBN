//
// Created by João Santos on 16/12/2020.
//

#include "hill.h"

#include <unordered_map>
#include <common/container_hasher.hpp>

namespace Common {
    void Operation::perform() const {
        switch (type) {
            case Add:
                graph->addEdge(edge.source, edge.target);
                break;
            case Delete:
                graph->removeEdge(edge.source, edge.target);
                break;
            case Reverse:
                graph->removeEdge(edge.source, edge.target);
                graph->addEdge(edge.target, edge.source);
                break;
        }
    }

    bool operator> (const Operation &a, const Operation &b) {
        return a.score > b.score;
    }

    bool operator< (const Operation &a, const double &b) {
        return a.score < b;
    }

    bool operator> (const double &b, const Operation &a) {
        return b > a.score;
    }

    bool operator> (const Operation &a, const double &b) {
        return a.score > b;
    }

    template<> double Score<LogLikelihood>::evaluate(Node node, const std::list<Node> &parents) const {
        std::unordered_map<std::vector<uint8_t>, std::vector<int>, ContainerHash<std::vector<uint8_t>>> counts;

        for (int i = 0; i < M; i++) {
            const int value = data[node+i*rvs];
            std::vector<uint8_t> parentLine(parents.size());
            int j = 0;

            for (const Node &parent : parents) {
                parentLine[j++] = data[parent+i*rvs];
            }

            if (counts.find(parentLine) == counts.end()) {
                counts[parentLine] = std::vector<int>(states[node]);
            }

            counts[parentLine][value]++;
        }

        double result = 0;

        for (auto &countPair : counts) {
            const std::vector<int> valueCounts = countPair.second;
            int totalCounts = 0;

            for (const int &count : valueCounts) totalCounts += count;

            for (const int &count : valueCounts) {
                if (count > 0) {
                    result += count * log(count / (double) totalCounts);
                }
            }
        }

        return result;
    }

    template <> double Score<LogLikelihood>::evaluate(const Digraph &graph) const {
        double score = 0.0;

        for (int node = 0; node < graph.nodes; node++) {
            score += evaluate(node, graph.getParents(node));
        }

        return score;
    }
}
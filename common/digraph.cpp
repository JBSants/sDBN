//
// Created by João Santos on 15/12/2020.
//

#include "digraph.h"

#include <queue>
#include <iostream>
#include <algorithm>

namespace Common {
    namespace Graph {
        void Digraph::addEdge(Node source, Node target) {
            inView[target].emplace_back(source);
        }

        void Digraph::removeEdge(Node source, Node target) {
            inView[target].remove(source);
        }

        bool Digraph::hasEdge(Node source, Node target) const {
            return std::any_of(inView[target].begin(), inView[target].end(), [source](int i) { return i == source; });
        }

        std::vector<Edge> Digraph::getEdges() const {
            std::vector<Edge> result;

            for (int i = 0; i < nodes; i++) {
                for (int parent : inView[i]) {
                    result.emplace_back((Edge) { parent, i });
                }
            }

            return result;
        }

        const std::list<Node> &Digraph::getParents(Node node) const {
            return inView[node];
        }

        int Digraph::getNodes() const {
            return nodes;
        }

        bool Digraph::hasPath(Node source, Node target) const {
            std::vector<bool> visited(nodes);
            std::queue<Node> q;
            q.push(target);

            while (!q.empty()) {
                Node &next = q.front();
                visited[next] = true;
                for (const Node &parent : inView[next]) {
                    if (parent == source) return true;
                    if (!visited[parent]) q.push(parent);
                }
                q.pop();
            }

            return false;
        }

        bool Digraph::reversalCausesCycle(Node source, Node target) const {
            std::vector<bool> visited(nodes);
            std::queue<Node> q;

            for (const Node &parent : inView[target]) {
                if (parent == source) continue;
                q.push(parent);
            }

            while (!q.empty()) {
                Node &next = q.front();
                visited[next] = true;
                for (const Node &parent : inView[next]) {
                    if (parent == source) return true;
                    if (!visited[parent]) q.push(parent);
                }
                q.pop();
            }

            return false;
        }

        void Digraph::writeDot(std::ostream &out) const {
            out << "digraph{" << std::endl;

            for (Node node = 0; node < nodes; node++) out << "\t" << node << "; " << std::endl;
            for (Node node = 0; node < nodes; node++) {
                for (Node parent : inView[node]) {
                    out << "\t" << parent << " -> " << node << ";" << std::endl;
                }
            }

            out << "}";
        }
    }
}

//
// Created by João Santos on 15/12/2020.
//

#ifndef CDBAYES_DIGRAPH_H
#define CDBAYES_DIGRAPH_H

#include <list>
#include <vector>
#include <iostream>

namespace Common {
    namespace Graph {
        typedef int Node;

        struct Edge {
            Node source;
            Node target;

            bool operator==(const Edge &other) const {
                return source == other.source && target == other.target;
            }
        };

        struct Digraph {
            int nodes;
            std::vector<std::list<Node>> inView;

        public:
            explicit Digraph(int nodes) : nodes(nodes), inView(nodes) {};

            void addEdge(Node source, Node target);
            void removeEdge(Node source, Node target);
            bool hasEdge(Node source, Node target) const;
            std::vector<Edge> getEdges() const;

            const std::list<Node> &getParents(Node node) const;
            int getNodes() const;

            bool hasPath(Node source, Node target) const;
            bool reversalCausesCycle(Node source, Node target) const;

            void writeDot(std::ostream &out) const;
        };

    }
}

#endif //CDBAYES_DIGRAPH_H

//
// Created by João Santos on 16/12/2020.
//

#ifndef CDBAYES_HILL_H
#define CDBAYES_HILL_H

#include "common/digraph.h"
#include "regularized_network/dataset.h"

namespace Common {
    using namespace Graph;

    enum OperationType {
        Add, Delete, Reverse
    };

    struct Operation {
        OperationType type;

        Digraph *graph;
        Edge edge;
        double score = -std::numeric_limits<double>::infinity();

    public:
        Operation(Digraph &graph, OperationType type, Edge edge, double score) : graph(&graph), type(type), edge(edge), score(score) {};
        Operation() = default;

        void perform() const;
        friend bool operator> (const Operation &a, const Operation &b);
        friend bool operator< (const Operation &a, const double &b);
        friend bool operator> (const double &b, const Operation &a);
        friend bool operator> (const Operation &a, const double &b);
    };

    enum ScoreType {
        LogLikelihood
    };

    struct TrainResult {
        double logLikelihood;
        Digraph &structure;
    };

    template <ScoreType T> struct Score {
        std::vector<uint8_t> data;
        int rvs;
        int M;
        std::vector<int> states;
    public:
         Score(const RegularizedNetwork::Dataset &dataset) : rvs(dataset.randomVariablesStates.size()),
                                                                        M(dataset.dataset->rows()),
                                                                        states(dataset.randomVariablesStates) {
            data = std::vector<uint8_t>(M*rvs);

            for (int i = 0; i < M; i++) {
                for (int j = 0; j < rvs; j++) {
                    data[j+i*rvs] = 0;
                    for (int s = 0; s < states[j]-1; s++) {
                        if ((*dataset.dataset)(i, dataset.randomVariablesIndices[j]+s) != 0) {
                            data[j+i*rvs] = s+1;
                        }
                    }
                }
            }
        }

        double evaluate(Node node, const std::list<Node> &parents) const;
        double evaluate(const Digraph &graph) const;
    };

    template<ScoreType T> double TrainRestricted(const std::vector<Edge> &allowed, const Score<T> &score, Digraph &graph, int maxParents = std::numeric_limits<int>::max()) {
        double logLikelihood = score.evaluate(graph);
        std::cout << "Directing  restricted to " << allowed.size() << " edges." << std::endl;

        for (int iter = 0; iter < 10000; iter++) {
            Operation bestOperation;

            //Test addition
            for (const Edge &edge : allowed) {
                if (graph.hasEdge(edge.source, edge.target) || graph.hasPath(edge.target, edge.source)) continue;

                std::list<Node> newParents = graph.getParents(edge.target);

                if (newParents.size() > maxParents - 1) continue;

                double improvement = -score.evaluate(edge.target, newParents);

                newParents.emplace_back(edge.source);

                improvement += score.evaluate(edge.target, newParents);

                if (improvement > bestOperation) bestOperation = Operation(graph, Add, edge, improvement);
            }

            //Test deletion
            for (int node = 0; node < graph.getNodes(); node++) {
                std::list<Node> parents = graph.getParents(node);
                double initialScore = -score.evaluate(node, parents);

                for (auto it = parents.begin(); it != parents.end();) {
                    Node parent = *it;
                    parents.erase(it++);

                    double improvement = initialScore + score.evaluate(node, parents);

                    if (improvement > bestOperation) bestOperation = Operation(graph, Delete, {parent, node}, improvement);

                    parents.insert(it, parent);
                }
            }

            //Test reversion
            for (int node = 0; node < graph.getNodes(); node++) {
                std::list<Node> parents = graph.getParents(node);
                double initialScore = -score.evaluate(node, parents);

                for (auto it = parents.begin(); it != parents.end();) {
                    Node parent = *it;

                    if (graph.reversalCausesCycle(parent, node)) {
                        it++;
                        continue;
                    }

                    parents.erase(it++);

                    double improvement = initialScore;

                    std::list<Node> sourceParents = graph.getParents(parent);

                    if (sourceParents.size() > maxParents - 1) continue;

                    improvement -= score.evaluate(parent, sourceParents);

                    sourceParents.emplace_back(node);

                    improvement += score.evaluate(parent, sourceParents);
                    improvement += score.evaluate(node, parents);

                    if (improvement > bestOperation) bestOperation = Operation(graph, Reverse, {parent, node}, improvement);

                    parents.insert(it, parent);
                }
            }

            if (bestOperation.score < 0.0001) break;

            bestOperation.perform();
            logLikelihood += bestOperation.score;
        }

        return logLikelihood;
    }

    template<ScoreType T> double TrainTimeseriesRestricted(const std::vector<Edge> &allowed, const Score<T> &score, const std::vector<int> &randomVariablesTimesteps, Digraph &graph, int maxParents = std::numeric_limits<int>::max()) {
        double logLikelihood = score.evaluate(graph);
        std::cout << "Directing restricted to " << allowed.size() << " edges." << std::endl;

        for (int iter = 0; iter < 10000; iter++) {
            Operation bestOperation;

            //Test addition (it is assumed that allowedEdges are valid)
            for (const Edge &edge : allowed) {
                if (graph.hasEdge(edge.source, edge.target) || graph.hasPath(edge.target, edge.source)) continue;

                std::list<Node> newParents = graph.getParents(edge.target);

                if (newParents.size() > maxParents - 1) continue;

                double improvement = -score.evaluate(edge.target, newParents);

                newParents.emplace_back(edge.source);

                improvement += score.evaluate(edge.target, newParents);

                if (improvement > bestOperation) bestOperation = Operation(graph, Add, edge, improvement);
            }

            //Test deletion
            for (int node = 0; node < graph.getNodes(); node++) {
                std::list<Node> parents = graph.getParents(node);
                double initialScore = -score.evaluate(node, parents);

                for (auto it = parents.begin(); it != parents.end();) {
                    Node parent = *it;
                    parents.erase(it++);

                    double improvement = initialScore + score.evaluate(node, parents);

                    if (improvement > bestOperation) bestOperation = Operation(graph, Delete, {parent, node}, improvement);

                    parents.insert(it, parent);
                }
            }

            //Test reversion
            for (int node = 0; node < graph.getNodes(); node++) {
                std::list<Node> parents = graph.getParents(node);
                double initialScore = -score.evaluate(node, parents);

                for (auto it = parents.begin(); it != parents.end();) {
                    Node parent = *it;

                    if (randomVariablesTimesteps[node] != randomVariablesTimesteps[parent]) {
                        it++;
                        continue;
                    }

                    if (graph.reversalCausesCycle(parent, node)) {
                        it++;
                        continue;
                    }

                    parents.erase(it++);

                    double improvement = initialScore;

                    std::list<Node> sourceParents = graph.getParents(parent);

                    if (sourceParents.size() > maxParents - 1) continue;

                    improvement -= score.evaluate(parent, sourceParents);

                    sourceParents.emplace_back(node);

                    improvement += score.evaluate(parent, sourceParents);
                    improvement += score.evaluate(node, parents);

                    if (improvement > bestOperation) bestOperation = Operation(graph, Reverse, {parent, node}, improvement);

                    parents.insert(it, parent);
                }
            }

            if (bestOperation.score < 0.0001) break;

            bestOperation.perform();
            logLikelihood += bestOperation.score;

            if (!(iter % 100)) {
                std::cout << "[Directing] Iteration " << iter << " (lik=" << logLikelihood << ")." << std::endl;
            }
        }

        return logLikelihood;
    }
}

#endif //CDBAYES_HILL_H

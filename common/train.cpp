//
// Created by João Santos on 17/12/2020.
//

#include "train.h"

#include <cmath>
#include <common/digraph.h>
#include <algorithm>

namespace Common {
    namespace Train {
        std::vector<double> GenerateLogGrid(double start, double end, int size) {
            std::vector<double> result;
            double step = std::exp((1 / (double) (size - 1)) * (std::log(start) - std::log(end)));

            result.reserve(size);

            for (int i = 0; i < size; i++) {
                result.emplace_back(end);
                end *= step;
            }

            return result;
        }

        std::vector<double> ReadGrid(std::istream &input) {
            unsigned int size;

            input >> size;

            std::vector<double> result(size);

            for (int i = 0; i < size; i++) {
                input >> result[i];
            }

            return result;
        }

        std::vector<double> ReadLogGridConfiguration(const std::string &configuration) {
            int size;
            double start, end;

            if (std::sscanf(configuration.data(), "%d,%lf,%lf", &size, &start, &end) == 3) {
                return GenerateLogGrid(start, end, size);
            }

            return {};
        }

        void ReadMaxEdgesConfiguration(const std::string &configuration, int &maxEdges, double &step, double &initialLambda, int &maxPathSize) {
            if (std::sscanf(configuration.data(), "%d,%lf,%lf,%d", &maxEdges, &step, &initialLambda, &maxPathSize) != 4) {
                std::cerr << "Max edges input malformed." << std::endl;
                std::exit(EXIT_FAILURE);
            }
        }

        void outputEdgesArray(std::ostream &output, const std::vector<Common::Graph::Edge> &edges, int nodes) {
            output << "{\"nodes\":" << nodes << ", \"edges\":[";

            for (auto &edge : edges) {
                output << "{\"source\":" << edge.source << ",\"target\":" << edge.target << "}";

                if (&edge != &edges.back()) output << ",";
            }

            output << "]}";
        }

        void OutputResult(std::ostream &output, double lambda, const std::vector<Common::Graph::Edge> &allowedEdges,
                          const Common::TrainResult *result, int nodes) {
            output << "{";

            output << "\"lambda\":" << lambda;
            output << ",\"skeleton\":";

            outputEdgesArray(output, allowedEdges, nodes);

            if (result) {
                output << ",\"directed\":{";

                output << "\"likelihood\":" << result->logLikelihood << ",\"structure\":";

                outputEdgesArray(output, result->structure.getEdges(), nodes);

                output << "}";
            }

            output << "}";
        }

        std::vector<Common::Graph::Edge> GetAllowedEdges(const RegularizedNetwork::Dataset &dataset, const RegularizedNetwork::PathTrainResult &betas, int l) {
            std::vector<Common::Graph::Edge> edges;

            for (int i = 0; i < dataset.randomVariablesStates.size(); i++) {
                const Eigen::MatrixXd &rvBetas = betas.getBetas(i, l);

                for (int j = 0; j < dataset.randomVariablesStates.size(); j++) {
                    if (rvBetas.block(0, dataset.randomVariablesIndices[j], rvBetas.rows(), dataset.randomVariablesStates[j]-1).any()) {
                        edges.emplace_back(Common::Graph::Edge {j, i});
                    }
                }
            }

            return edges;
        }

        std::vector<int> FindClosestIndicesForLambdas(const std::vector<double> &originalLambdas, const std::vector<double> &newLambdas) {
            std::vector<int> closestIndices(newLambdas.size());

            for (int i = 0; i < newLambdas.size(); i++) {
                double minDistance = std::numeric_limits<double>::max();

                for (int j = 0; j < originalLambdas.size(); j++) {
                    double distance = abs(originalLambdas[j] - newLambdas[i]);

                    if (distance <= minDistance) {
                        minDistance = distance;
                        closestIndices[i] = j;
                    }
                }
            }

            return closestIndices;
        }

        Graph::Digraph FilterNotAllowedEdges(const Graph::Digraph &graph, const std::vector<Common::Graph::Edge> &allowedEdges) {
            Graph::Digraph result = graph;

            //TODO: Otimizar loop O(E^2)
            for (auto edge : graph.getEdges()) {
                if (std::find(allowedEdges.begin(), allowedEdges.end(), edge) == allowedEdges.end()) {
                    result.removeEdge(edge.source, edge.target);
                }
            }

            return result;
        }
    }
}
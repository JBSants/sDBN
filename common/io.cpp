//
// Created by João Santos on 15/03/2021.
//

#include "io.h"
#include "digraph.h"
#include <memory>

namespace Common {
    namespace IO {
        void serializeSparseMatrix(std::ostream &output, const Eigen::SparseMatrix<double> &matrix) {
            assert(matrix.isCompressed());
            int nnz = matrix.nonZeros();
            const int cols = matrix.cols();
            const int rows = matrix.rows();

            output.write((char *) &cols,sizeof(int));
            output.write((char *) &rows,sizeof(int));
            output.write((char *) &nnz, sizeof(int));
            output.write((char *) matrix.valuePtr(), nnz*sizeof(double));
            output.write((char *) matrix.innerIndexPtr(), nnz*sizeof(int));
            output.write((char *) matrix.outerIndexPtr(), matrix.cols()*sizeof(int));
        }

        Eigen::SparseMatrix<double> unserializeSparseMatrix(std::istream &input) {
            int cols, rows, nnz;

            input.read((char *) &cols, sizeof(int));
            input.read((char *) &rows, sizeof(int));
            input.read((char *) &nnz, sizeof(int));

            Eigen::SparseMatrix<double> matrix(rows, cols);

            matrix.reserve(nnz);
            matrix.resizeNonZeros(nnz);

            input.read((char *) matrix.valuePtr(), nnz*sizeof(double));
            input.read((char *) matrix.innerIndexPtr(), nnz*sizeof(int));
            input.read((char *) matrix.outerIndexPtr(), cols*sizeof(int));
            matrix.outerIndexPtr()[cols] = nnz;

            return matrix;
        }

        void serializeDoubleVector(std::ostream &output, const std::vector<double> &vec) {
            int size = vec.size();
            output.write((char *) &size, sizeof(int));

            for (const double &a : vec) {
                output.write((char *) &a, sizeof(double));
            }
        }

        std::vector<double> unserializeDoubleVector(std::istream &input) {
            int size;
            std::vector<double> result;

            input.read((char *) &size, sizeof(int));

            result.resize(size);
            input.read((char *) result.data(), sizeof(double)*size);

            return result;
        }

        void SerializePathTrainResult(std::ostream &output, const RegularizedNetwork::PathTrainResult &result) {
            int size = result.getRVCount();
            output.write((char *) &size, sizeof(int));

            serializeDoubleVector(output, result.getLambdas());

            for (int rv = 0; rv < result.getRVCount(); rv++) {
                for (int idx = 0; idx < result.getLambdas().size(); idx++) {
                    serializeSparseMatrix(output, result.getBetas(rv, idx));
                }
            }
        }

        RegularizedNetwork::FixedPathTrainResult UnserializePathTrainResult(std::istream &input) {
            int rvs;
            input.read((char *) &rvs, sizeof(int));

            std::vector<double> lambdas = unserializeDoubleVector(input);

            std::vector<Eigen::SparseMatrix<double>> pathBetas;

            for (int rv = 0; rv < rvs; rv++) {
                for (int idx = 0; idx < lambdas.size(); idx++) {
                    pathBetas.push_back(unserializeSparseMatrix(input));
                }
            }

            return RegularizedNetwork::FixedPathTrainResult(std::move(lambdas), std::move(pathBetas));
        }

        void SerializeDigraph(std::ostream &output, const Graph::Digraph &digraph) {
            output.write((char *) &digraph.nodes, sizeof(int));

            for (auto &adj : digraph.inView) {
                int inDegree = adj.size();
                output.write((char *) &inDegree, sizeof(int));

                for (Graph::Node parent : adj) {
                    output.write((char *) &parent, sizeof(Graph::Node));
                }
            }
        }

        Graph::Digraph UnserializeDigraph(std::istream &input) {
            int nodesCount;

            input.read((char *) &nodesCount, sizeof(int));

            Graph::Digraph result(nodesCount);

            for (int i = 0; i < nodesCount; i++) {
                int inDegree;

                input.read((char *) &inDegree, sizeof(int));

                for (int j = 0; j < inDegree; j++) {
                    int adj;

                    input.read((char *) &adj, sizeof(Graph::Node));

                    result.addEdge(adj, i);
                }
            }

            return result;
        }

        void SerializeHillClimbingHotStart(std::ostream &output, const HillClimbingHotStart &hotstart) {
            serializeDoubleVector(output, hotstart.lambdas);

            for (auto &digraph : hotstart.directed) {
                SerializeDigraph(output, digraph);
            }
        }

        HillClimbingHotStart UnserializeHillClimbingHotStart(std::istream &input) {
            std::vector<double> lambdas = unserializeDoubleVector(input);
            std::vector<Graph::Digraph> directed;

            directed.reserve(lambdas.size());

            for (auto lambda : lambdas) {
                directed.push_back(UnserializeDigraph(input));
            }

            return { std::move(lambdas), std::move(directed) };
        }

        void SerializeHotStart(std::ostream &output, const HotStart &hotstart) {
            SerializePathTrainResult(output, hotstart.regularizedNetworkHotStart);
            SerializeHillClimbingHotStart(output, hotstart.hillClimbingHotStart);
        }

        HotStart UnserializeHotStart(std::istream &input) {
            return { UnserializePathTrainResult(input), UnserializeHillClimbingHotStart(input) };
        }
    }
}
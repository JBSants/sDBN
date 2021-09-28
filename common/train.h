//
// Created by João Santos on 17/12/2020.
//

#ifndef CDBAYES_TRAIN_H
#define CDBAYES_TRAIN_H

#include <iostream>
#include <vector>
#include <common/digraph.h>
#include <hillclimber/hill.h>
#include <regularized_network/PathTrainResult.h>

namespace Common {
    namespace Train {
        std::vector<double> GenerateLogGrid(double start, double end, int size);

        std::vector<double> ReadGrid(std::istream &input);

        std::vector<double> ReadLogGridConfiguration(const std::string &configuration);
        void ReadMaxEdgesConfiguration(const std::string &configuration, int &maxEdges, double &step, double &initialLambda, int &maxPathSize);

        void OutputResult(std::ostream &output, double lambda, const std::vector<Common::Graph::Edge> &allowedEdges,
                          const Common::TrainResult *result, int nodes);

        std::vector<Common::Graph::Edge>
        GetAllowedEdges(const RegularizedNetwork::Dataset &dataset, const RegularizedNetwork::PathTrainResult &betas,
                        int l);

        std::vector<int> FindClosestIndicesForLambdas(const std::vector<double> &originalLambdas, const std::vector<double> &newLambdas);
        Graph::Digraph FilterNotAllowedEdges(const Graph::Digraph &graph, const std::vector<Common::Graph::Edge> &allowedEdges);
    }
}

#endif //CDBAYES_TRAIN_H

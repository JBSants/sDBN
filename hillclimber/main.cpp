//
// Created by João Santos on 16/12/2020.
//

#include "common/cxxopts.hpp"
#include "regularized_network/dataset.h"
#include "common/digraph.h"
#include "hill.h"

#include <fstream>
#include <string>

int main(int argc, char **argv) {
    cxxopts::Options options("hill", "Directs a dynamic Bayesian network restricted to a set of edges.");

    options.add_options()
            ("d,dataset", "Dataset file name", cxxopts::value<std::string>())
            ("i,input", "Input file name", cxxopts::value<std::string>());

    auto result = options.parse(argc, argv);

    std::ifstream datasetFile, inputFile;

    datasetFile.open(result["dataset"].as<std::string>());
    inputFile.open(result["input"].as<std::string>());

    if (!datasetFile.is_open() || !inputFile.is_open()) return EXIT_FAILURE;

    auto dataset = RegularizedNetwork::ReadDataset(datasetFile);
    int edges;

    inputFile >> edges;
    std::vector<Common::Graph::Edge> allowed;

    for (int i = 0; i < edges; i++) {
        Common::Graph::Edge edge{};

        inputFile >> edge.source >> edge.target;

        allowed.emplace_back(edge);
    }

    Common::Score<Common::ScoreType::LogLikelihood> score(dataset);
    Common::Digraph resultGraph(dataset.randomVariablesIndices.size());
    Common::TrainRestricted(allowed, score, resultGraph);

    resultGraph.writeDot(std::cerr);
}
//
// Created by João Santos on 12/03/2021.
//
#include <iostream>
#include <fstream>
#include <common/io.h>
#include "common/cxxopts.hpp"
#include "common/train.h"
#include "regularized_network/dataset.h"
#include "train.h"
#include "regularized_network/local/BNPathTrainer.h"

#ifdef MPI_BUILD
    #include <mpi.h>
    #include "regularized_network/distributed/BNPathTrainer.h"
#endif

int main(int argc, char **argv) {
#ifdef MPI_BUILD
    MPI_Init(&argc, &argv);
#endif
    cxxopts::Options options("bntrain", "Trains bayesian networks using sparsebn approximations.");

    options.add_options()
            ("d,dataset", "Dataset file name", cxxopts::value<std::string>())
            ("b,binary-dataset", "Binary dataset format", cxxopts::value<bool>()->default_value("false"))
            ("o,output", "Output file name.", cxxopts::value<std::string>())

            ("H,hot-start", "Hot start solution file.", cxxopts::value<std::string>())
            ("W,output-hot-start", "Output hot start solution file.", cxxopts::value<std::string>())

            ("Y,no-hill", "Disable direct output using hill climbing.", cxxopts::value<bool>()->default_value("false"))
            ("p,max-parents", "Hill climbing max parents", cxxopts::value<int>())

            ("E,epsilon", "Epsilon parameter for lbfgs optimization.", cxxopts::value<double>())

            ("g,log-grid", "Train over a log grid of lambda values. Specify a string of form min,max,size", cxxopts::value<std::string>())

            ("e,eras", "Number of eras to train in distributed mode.", cxxopts::value<int>()->default_value("1"))
            ("y,gamma", "Gamma parameter for adaptative weights.", cxxopts::value<double>()->default_value("1.0"));

    cxxopts::ParseResult result = options.parse(argc, argv);
    std::istream *input;
    std::ostream *output;
    std::ifstream inputFile;
    std::ofstream outputFile;

    if (result["dataset"].count() > 0) {
        // Handle read dataset from file
        inputFile.open(result["dataset"].as<std::string>());
        input = &inputFile;
    } else {
        // Handle read dataset from stdin
        input = &std::cin;
    }

    if (result["output"].count() > 0) {
        // Handle write output to file
        outputFile.open(result["output"].as<std::string>());
        output = &outputFile;
    } else {
        // Handle write output to stdout
        output = &std::cout;
    }

    std::vector<double> lambdas;

    if (result["log-grid"].count() <= 0) {
        lambdas = Common::Train::ReadGrid(std::cin);
    } else {
        lambdas = Common::Train::ReadLogGridConfiguration(result["log-grid"].as<std::string>());
    }

    auto dataset = RegularizedNetwork::ReadDataset(*input);
    int mpiSize, mpiRank;

    int rvs = dataset.randomVariablesIndices.size();
    Eigen::MatrixXd weights(rvs, rvs);

    if (result["epsilon"].count() <= 0) {
        BNTrain::GetAdaptativeWeights(dataset, weights, result["gamma"].as<double>());
    } else {
        BNTrain::GetAdaptativeWeights(dataset, weights, result["gamma"].as<double>(), result["epsilon"].as<double>());
    }
    std::unique_ptr<Common::HotStart> hotStart;

    if (result["hot-start"].count() > 0) {
        std::ifstream hotStartFile(result["hot-start"].as<std::string>());

        if (hotStartFile.is_open()) {
            hotStart = std::make_unique<Common::HotStart>(
                    Common::IO::UnserializeHotStart(hotStartFile));
            hotStartFile.close();
        }
    }

#ifdef MPI_BUILD
    MPI_Comm_size(MPI_COMM_WORLD, &mpiSize);
    MPI_Comm_rank(MPI_COMM_WORLD, &mpiRank);
#else
    mpiSize = 1;
    mpiRank = 0;
#endif

    std::unique_ptr<RegularizedNetwork::PathTrainResult> betas;

    if (mpiSize > 1) {
#ifdef MPI_BUILD
        betas = RegularizedNetwork::Distributed::BNPathTrainer(dataset, weights, "").train(lambdas, result["eras"].as<int>());
#endif
    } else {
        if (hotStart) {
            betas = RegularizedNetwork::Local::BNPathTrainer(dataset, weights).train(lambdas, hotStart->regularizedNetworkHotStart);
        } else {
            betas = RegularizedNetwork::Local::BNPathTrainer(dataset, weights).train(lambdas, 1);
        }
    }


    std::string outputHotFilename;

    if (result["output-hot-start"].count() > 0) {
        std::ofstream hotStartFile;

        outputHotFilename = result["output-hot-start"].as<std::string>();

        hotStartFile.open(outputHotFilename);

        if (hotStartFile.is_open()) {
            if (!hotStart) {
                hotStart = std::make_unique<Common::HotStart>(*betas, Common::HillClimbingHotStart());
            } else {
                hotStart->regularizedNetworkHotStart = *betas;
            }

            Common::IO::SerializeHotStart(hotStartFile, *hotStart);
        }
    }

    if (result["no-hill"].as<bool>()) {
        BNTrain::OutputPath(*output, dataset, *betas);
    } else {
        int maxParents = result["max-parents"].count() > 0 ? result["max-parents"].as<int>() : std::numeric_limits<int>::max();
        BNTrain::DirectAndOutputPath(*output, dataset, *betas, maxParents, hotStart.get(), !outputHotFilename.empty() ? &outputHotFilename : nullptr);
    }

#ifdef MPI_BUILD
    MPI_Finalize();
#endif
}

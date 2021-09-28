//
// Created by João Santos on 12/03/2021.
//

#include "train.h"
#include <fstream>
#include <chrono>
#include <common/io.h>
#include "adaptative/weights.h"
#include "hillclimber/hill.h"
#include "common/train.h"

#ifdef MPI_BUILD
    #include <mpi.h>
#endif

namespace BNTrain {
    void GetAdaptativeWeights(const RegularizedNetwork::Dataset &dataset, Eigen::MatrixXd &weights, const double gamma, const double epsilon) {
        int mpiRank = 0;

#ifdef MPI_BUILD
        MPI_Comm_rank(MPI_COMM_WORLD, &mpiRank);
#endif

        const int rvs = dataset.randomVariablesStates.size();
        std::string cacheFilename = ".bntrain_cache." + dataset.checksum;
        std::ifstream cacheWeights(cacheFilename, std::ios::binary);

        std::cerr << "Cache filename: " << cacheFilename << std::endl;

        if (cacheWeights.is_open()) {
            for (int i = 0; i < rvs*rvs; i++) {
                cacheWeights.read((char *) &weights(i), sizeof(double));
            }

            cacheWeights.close();
        } else {
            DiscoverAdaptativeWeights(dataset, weights, gamma, epsilon);

            if (mpiRank == 0) {
                std::ofstream cache;
                cache.open(cacheFilename, std::ios::binary);

                for (int i = 0; i < rvs * rvs; i++) {
                    cache.write((char *) &weights(i), sizeof(double));
                }

                cache.close();
            }
        }
    }

    void DirectAndOutputPath(std::ostream &output, const RegularizedNetwork::Dataset &dataset, const RegularizedNetwork::PathTrainResult &betas, int maxParents, Common::HotStart *hotStart, std::string *hotstartFile) {
        output << "[";

        const std::vector<double> &path = betas.getLambdas();
        Common::Score<Common::ScoreType::LogLikelihood> score(dataset);
        int pathSize = path.size();
        std::vector<int> closestIndices;

        std::unique_ptr<Common::HotStart> newHotstart;

        if (hotStart) {
            closestIndices = Common::Train::FindClosestIndicesForLambdas(path, hotStart->hillClimbingHotStart.lambdas);

            if (hotstartFile) {
                newHotstart = std::make_unique<Common::HotStart>(Common::HotStart(hotStart->regularizedNetworkHotStart, Common::HillClimbingHotStart()));
            }
        }

        for (int l = 0; l < pathSize; l++) {
            std::cerr << "\nDirecting network " << l << " out of " << pathSize << std::endl;

            auto edges = Common::Train::GetAllowedEdges(dataset, betas, l);
            Common::Graph::Digraph result(dataset.randomVariablesIndices.size());

            if (hotStart && !closestIndices.empty()) {
                result = Common::Train::FilterNotAllowedEdges(hotStart->hillClimbingHotStart.directed[closestIndices[l]], edges);
            }

            auto start = std::chrono::high_resolution_clock::now();
            double resultLikelihood = Common::TrainRestricted(edges, score, result, maxParents);
            auto end = std::chrono::high_resolution_clock::now();

            std::chrono::duration<double> elapsed = end - start;
            std::cerr << "It took " << elapsed.count() << " s\n";

            Common::TrainResult trainResult = { resultLikelihood, result };

            Common::Train::OutputResult(output, path[l], edges, &trainResult, dataset.randomVariablesIndices.size());

            if (hotstartFile && hotStart) {
                std::ofstream outputHotstartFile(*hotstartFile);
                newHotstart->hillClimbingHotStart.lambdas.push_back(path[l]);
                newHotstart->hillClimbingHotStart.directed.push_back(std::move(result));

                Common::IO::SerializeHotStart(outputHotstartFile, *newHotstart);

                outputHotstartFile.close();
            }

            if (l < path.size() - 1) output << ",";
        }

        output << "]";
    }

    void OutputPath(std::ostream &output, const RegularizedNetwork::Dataset &dataset, const RegularizedNetwork::PathTrainResult &betas) {
        output << "[";

        const std::vector<double> &path = betas.getLambdas();

        for (int l = 0; l < path.size(); l++) {
            Common::Train::OutputResult(output, path[l], Common::Train::GetAllowedEdges(dataset, betas, l), nullptr, dataset.randomVariablesIndices.size());

            if (l < path.size() - 1) output << ",";
        }

        output << "]";
    }
}

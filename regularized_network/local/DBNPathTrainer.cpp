//
// Created by João Santos on 28/12/2020.
//

#include <iostream>
#include <regularized_network/DynamicPathTrainResult.h>
#include "DBNPathTrainer.h"
#include "regularized_network/DBNNodeTrainer.h"
#include "common/train.h"

namespace RegularizedNetwork {
    namespace Local {
        std::unique_ptr<PathTrainResult> DBNPathTrainer::train(const std::vector<double> &lambdas, int eras) {
            const int rvs = dataset.randomVariablesStates.size();
            std::unique_ptr<FixedPathTrainResult> trainResult = std::make_unique<FixedPathTrainResult>(dataset,
                                                                                                       lambdas);

            for (int i = dataset.stubTimesteps * ((rvs - dataset.staticRandomVariables) / dataset.timesteps);
                 i < rvs; i++) {
                VectorXd weightsRow = weights.row(i);
                RegularizedNetwork::DBNNodeTrainer trainer(dataset, i, rvs, weightsRow);
                auto trainedPath = trainer.trainPath(lambdas);

                for (int j = 0; j < lambdas.size(); j++) trainResult->setBetas(i, j, std::move(trainedPath[j]));
            }

            return trainResult;
        }

        std::unique_ptr<PathTrainResult> DBNPathTrainer::train(const std::vector<double> &lambdas, const FixedPathTrainResult &hotStart) {
            const int rvs = dataset.randomVariablesStates.size();
            std::unique_ptr<FixedPathTrainResult> trainResult = std::make_unique<FixedPathTrainResult>(dataset,
                                                                                                       lambdas);
            std::vector<int> closestIndices = Common::Train::FindClosestIndicesForLambdas(hotStart.getLambdas(), lambdas);

            for (int i = dataset.stubTimesteps * ((rvs - dataset.staticRandomVariables) / dataset.timesteps);
                 i < rvs; i++) {
                VectorXd weightsRow = weights.row(i);
                RegularizedNetwork::DBNNodeTrainer trainer(dataset, i, rvs, weightsRow);
                std::vector<Eigen::SparseMatrix<double>> trainedPath;

                for (int j = 0; j < lambdas.size(); j++) {
                    double lambda = lambdas[j];
                    trainer.betas = hotStart.getBetas(i, closestIndices[j]);
                    trainer.train(lambda, DEFAULT_MAX_CD_ITERS);
                    trainedPath.emplace_back(trainer.betas.sparseView());
                    std::cerr << "[Node " << i << "] Lambda " << lambda << " complete" << std::endl;
                }

                for (int j = 0; j < lambdas.size(); j++) trainResult->setBetas(i, j, std::move(trainedPath[j]));
            }

            return trainResult;
        }


        int DBNPathTrainer::countInEdges(const Eigen::MatrixXd &betas) {
            const int rvs = dataset.randomVariablesStates.size();
            int result = 0;

            for (int i = 0; i < rvs; i++) {
                result += betas.middleCols(dataset.randomVariablesIndices[i], dataset.randomVariablesStates[i]-1).any() ? 1 : 0;
            }

            return result;
        }

        std::unique_ptr<PathTrainResult> DBNPathTrainer::train(int maxEdges, double initialLambda, double step, int maxLambda) {
            const int rvs = dataset.randomVariablesStates.size();
            std::unique_ptr<DynamicPathTrainResult> trainResult = std::make_unique<DynamicPathTrainResult>(dataset);
            std::vector<DBNNodeTrainer> trainers;
            std::vector<Eigen::VectorXd> weightVectors;
            const int firstRv = dataset.stubTimesteps * ((rvs - dataset.staticRandomVariables) / dataset.timesteps);
            const int nonStubRvs = rvs - firstRv;

            trainers.reserve(nonStubRvs);
            weightVectors.reserve(nonStubRvs);
            for (int i = firstRv; i < rvs; i++) {
                weightVectors.emplace_back(weights.row(i));
                trainers.emplace_back(DBNNodeTrainer(dataset, i, rvs, weightVectors[i % nonStubRvs]));
            }

            double lambda = initialLambda;

            for (int j = 0; j < maxLambda; j++) {
                int selectedEdges = 0;

                trainResult->addLambda(lambda);

                for (int i = firstRv; i < rvs; i++) {
                    trainers[i % nonStubRvs].train(lambda, DEFAULT_MAX_CD_ITERS);
                    selectedEdges += countInEdges(trainers[i % nonStubRvs].betas);
                    trainResult->setBetas(i, lambda, trainers[i % nonStubRvs].betas.sparseView());
                }

                if (selectedEdges >= maxEdges) break;

                lambda *= step;
            }

            return trainResult;
        }
    }
}

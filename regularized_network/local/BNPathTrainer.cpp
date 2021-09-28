//
// Created by João Santos on 27/12/2020.
//

#include "BNPathTrainer.h"
#include "regularized_network/BNNodeTrainer.h"
#include "regularized_network/FixedPathTrainResult.h"
#include "regularized_network/DynamicPathTrainResult.h"
#include <memory>
#include <iostream>
#include <common/train.h>

namespace RegularizedNetwork {
    namespace Local {
        std::unique_ptr<PathTrainResult> BNPathTrainer::train(const std::vector<double> &lambdas, int eras) {
            const int rvs = dataset.randomVariablesStates.size();
            std::unique_ptr<FixedPathTrainResult> trainResult = std::make_unique<FixedPathTrainResult>(dataset, lambdas);

            for (int i = 0; i < rvs; i++) {
                VectorXd weightsRow = weights.row(i);
                RegularizedNetwork::BNNodeTrainer trainer(dataset, i, rvs, weightsRow);
                auto trainedPath = trainer.trainPath(lambdas);

                for (int j = 0; j < lambdas.size(); j++) trainResult->setBetas(i, j, std::move(trainedPath[j]));
            }

            return trainResult;
        }

        int BNPathTrainer::countInEdges(const Eigen::MatrixXd &betas) {
            const int rvs = dataset.randomVariablesStates.size();
            int result = 0;

            for (int i = 0; i < rvs; i++) {
                result += betas.middleCols(dataset.randomVariablesIndices[i], dataset.randomVariablesStates[i]-1).any() ? 1 : 0;
            }

            return result;
        }

        std::unique_ptr<PathTrainResult> BNPathTrainer::train(int maxEdges, double initialLambda, double step, int maxLambdas) {
            const int rvs = dataset.randomVariablesStates.size();
            std::unique_ptr<DynamicPathTrainResult> trainResult = std::make_unique<DynamicPathTrainResult>(dataset);
            std::vector<BNNodeTrainer> trainers;
            std::vector<Eigen::Ref<const VectorXd>> weightVectors;

            trainers.reserve(rvs);
            weightVectors.reserve(rvs);
            for (int i = 0; i < rvs; i++) {
                weightVectors.emplace_back(weights.row(i));
                trainers.emplace_back(BNNodeTrainer(dataset, i, rvs, weightVectors[i]));
            }

            double lambda = initialLambda;

            for (int j = 0; j < maxLambdas; j++) {
                int selectedEdges = 0;

                trainResult->addLambda(lambda);

                for (int i = 0; i < rvs; i++) {
                    trainers[i].train(lambda, DEFAULT_MAX_CD_ITERS);
                    selectedEdges += countInEdges(trainers[i].betas);
                    trainResult->setBetas(i, lambda, trainers[i].betas.sparseView());
                }

                if (selectedEdges >= maxEdges) break;

                lambda *= step;
            }

            return trainResult;
        }

        std::unique_ptr<PathTrainResult> BNPathTrainer::train(const std::vector<double> &lambdas, const FixedPathTrainResult &hotStart) {
            const int rvs = dataset.randomVariablesStates.size();
            std::unique_ptr<FixedPathTrainResult> trainResult = std::make_unique<FixedPathTrainResult>(dataset,
                                                                                                       lambdas);
            std::vector<int> closestIndices = Common::Train::FindClosestIndicesForLambdas(hotStart.getLambdas(), lambdas);

            for (int i = 0; i < rvs; i++) {
                VectorXd weightsRow = weights.row(i);
                RegularizedNetwork::BNNodeTrainer trainer(dataset, i, rvs, weightsRow);
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
    }
}

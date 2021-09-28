//
// Created by João Santos on 21/11/2020.
//

#ifndef CDBAYES_DISTRIBUTEDPATHTRAINER_H
#define CDBAYES_DISTRIBUTEDPATHTRAINER_H
#include "regularized_network/dataset.h"
#include <Eigen/Dense>
#include <mpi.h>
#include <iostream>
#include <queue>
#include <functional>
#include "../../generated/cdbayes_state_generated.h"
#include "Worker.h"
#include "../PathTrainResult.h"
#include "../PathTrainer.h"

using Eigen::MatrixXd;
namespace RegularizedNetwork {
    namespace Distributed {
        class BNPathTrainer : public RegularizedNetwork::PathTrainer {
        protected:
            const Dataset &dataset;
            const Eigen::MatrixXd &weights;
            int rvs;
            std::string stateFilename;
            CDBayesStateT state;
            bool noState;

            void recoverState(std::queue<int> &pool, PathTrainResult &result,
                              const std::vector<double> &lambdas);

            void writeState();

            void updateStateBetas(int rv, int updatedPathIdx,
                                  const Eigen::SparseMatrix<double> &betas,
                                  const std::vector<double> &lambdas);

            void updateStateEra();

            void clearState(const std::vector<double> &lambdas);

            void printState();

            static std::vector<Triplet> packBetas(const Eigen::SparseMatrix<double> &betas);

            virtual void managePool(PathTrainResult &result, int eras, int size,
                                    const std::vector<double> &lambdas);

            virtual void spawnWorker();

        public:
            int eraIterations = 5;

            BNPathTrainer(const Dataset &dataset, const Eigen::MatrixXd &weights, const std::string &stateFilename = "");

            std::unique_ptr<PathTrainResult> train(const std::vector<double> &lambdas, int eras) override;
            std::unique_ptr<PathTrainResult> train(int maxEdges, double initialLambda, double step, int maxLambda) override {
                return nullptr;
            };
        };
    }
}

#endif //CDBAYES_DISTRIBUTEDPATHTRAINER_H

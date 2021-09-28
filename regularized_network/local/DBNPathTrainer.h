//
// Created by João Santos on 28/12/2020.
//

#ifndef CDBAYES_DBNPATHTRAINER_H
#define CDBAYES_DBNPATHTRAINER_H

#include "regularized_network/PathTrainer.h"
#include "regularized_network/FixedPathTrainResult.h"

namespace RegularizedNetwork {
    namespace Local {
        class DBNPathTrainer : public RegularizedNetwork::PathTrainer {
            const TimeseriesDataset &dataset;
            const Eigen::MatrixXd &weights;

            int countInEdges(const Eigen::MatrixXd &betas);
        public:
            DBNPathTrainer(const TimeseriesDataset &dataset, const Eigen::MatrixXd &weights) : dataset(dataset), weights(weights) {};

            std::unique_ptr<PathTrainResult> train(const std::vector<double> &lambdas, int eras) override;
            std::unique_ptr<PathTrainResult> train(const std::vector<double> &lambdas, const FixedPathTrainResult &hotStart);
            std::unique_ptr<PathTrainResult> train(int maxEdges, double initialLambda, double step, int maxLambda) override;
        };
    }
}

#endif //CDBAYES_DBNPATHTRAINER_H

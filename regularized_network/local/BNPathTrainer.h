//
// Created by João Santos on 27/12/2020.
//

#ifndef CDBAYES_LOCALPATHTRAINER_H
#define CDBAYES_LOCALPATHTRAINER_H

#include "regularized_network/PathTrainer.h"
#include "regularized_network/FixedPathTrainResult.h"
#include <Eigen/Dense>

namespace RegularizedNetwork {
   namespace Local {
       class BNPathTrainer : public RegularizedNetwork::PathTrainer {
       private:
           const Dataset &dataset;
           const Eigen::MatrixXd &weights;

           int countInEdges(const Eigen::MatrixXd &betas);
       public:
           BNPathTrainer(const Dataset &dataset, const Eigen::MatrixXd &weights) : dataset(dataset), weights(weights) {};

           std::unique_ptr<PathTrainResult> train(const std::vector<double> &lambdas, int eras) override;
           std::unique_ptr<PathTrainResult> train(const std::vector<double> &lambdas, const FixedPathTrainResult &hotStart);
           std::unique_ptr<PathTrainResult> train(int maxEdges, double initialLambda, double step, int maxLambda) override;
       };
   }
}

#endif //CDBAYES_LOCALPATHTRAINER_H

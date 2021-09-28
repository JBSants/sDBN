//
// Created by João Santos on 12/03/2021.
//

#ifndef SPARSEDBN_TRAIN_H
#define SPARSEDBN_TRAIN_H

#include <iostream>
#include <vector>
#include "regularized_network/dataset.h"
#include "regularized_network/PathTrainResult.h"
#include <Eigen/Dense>
#include <common/HotStart.h>

namespace BNTrain {
    void GetAdaptativeWeights(const RegularizedNetwork::Dataset &dataset, Eigen::MatrixXd &weights, double gamma = 1.0, double epsilon = 1e-5);

    void DirectAndOutputPath(std::ostream &output, const RegularizedNetwork::Dataset &dataset, const RegularizedNetwork::PathTrainResult &betas, int maxParents, Common::HotStart *hotStart, std::string *hotstartFile);

    void OutputPath(std::ostream &output, const RegularizedNetwork::Dataset &dataset, const RegularizedNetwork::PathTrainResult &betas);
}

#endif //SPARSEDBN_TRAIN_H

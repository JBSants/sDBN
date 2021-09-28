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

namespace DBNTrain {
    void GetAdaptativeWeights(const RegularizedNetwork::TimeseriesDataset &dataset, Eigen::MatrixXd &weights, double gamma = 1.0, double epsilon = 1e-5);

    void DirectAndOutputPath(std::ostream &output, const RegularizedNetwork::TimeseriesDataset &dataset, const RegularizedNetwork::PathTrainResult &betas, int maxParents, Common::HotStart *hotStart, std::string *hotstartFile);

    void OutputPath(std::ostream &output, const RegularizedNetwork::TimeseriesDataset &dataset, const RegularizedNetwork::PathTrainResult &betas);
}

#endif //SPARSEDBN_TRAIN_H

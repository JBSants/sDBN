//
// Created by João Santos on 15/11/2020.
//

#ifndef CDBAYES_CDESCENT_H
#define CDBAYES_CDESCENT_H
#include <memory>
#include <Eigen/Eigen>
#include "regularized_network/dataset.h"

void DiscoverAdaptativeWeights(const RegularizedNetwork::Dataset &dataset, Eigen::MatrixXd &weights, double gamma = 1, double epsilon = 1e-5);
double FindMaxLambda(const RegularizedNetwork::Dataset &dataset, const Eigen::MatrixXd &weights);

#endif //CDBAYES_CDESCENT_H

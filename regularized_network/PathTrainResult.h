//
// Created by João Santos on 26/12/2020.
//

#ifndef CDBAYES_PATHTRAINRESULT_H
#define CDBAYES_PATHTRAINRESULT_H

#include <Eigen/Sparse>
#include <vector>
#include <unordered_map>

#include "dataset.h"

namespace RegularizedNetwork {
    class PathTrainResult {
    public:
        virtual ~PathTrainResult() = default;

        virtual void setBetas(int rv, int idx, Eigen::SparseMatrix<double> &&betas) = 0;
        virtual void setBetas(int rv, double lambda, Eigen::SparseMatrix<double> &&betas) = 0;

        virtual const Eigen::SparseMatrix<double> &getBetas(int rv, int idx) const = 0;
        virtual const Eigen::SparseMatrix<double> &getBetas(int rv, double lambda) const = 0;

        virtual const std::vector<double> &getLambdas() const = 0;
        virtual const int getRVCount() const = 0;
    };
}


#endif //CDBAYES_PATHTRAINRESULT_H

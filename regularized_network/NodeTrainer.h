//
// Created by João Santos on 21/11/2020.
//

#ifndef CDBAYES_NODETRAINER_H
#define CDBAYES_NODETRAINER_H

#include "dataset.h"

namespace RegularizedNetwork {
    class NodeTrainer {
    protected:
        int node;
        int V;
        const Dataset &dataset;

    public:
        NodeTrainer(const Dataset &dataset, int node, int V);
        ~NodeTrainer() = default;

        virtual void train(double lambda, int iters) = 0;
        virtual std::vector<Eigen::SparseMatrix<double>> trainPath(const std::vector<double> &lambdas) = 0;
    };
}


#endif //CDBAYES_NODETRAINER_H

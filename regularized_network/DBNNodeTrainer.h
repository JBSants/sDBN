//
// Created by João Santos on 29/11/2020.
//

#ifndef CDBAYES_DBNNODETRAINER_H
#define CDBAYES_DBNNODETRAINER_H

#include "BNNodeTrainer.h"
#include "dataset.h"


namespace RegularizedNetwork {
    class DBNNodeTrainer : public BNNodeTrainer {
        int timesteps;
        int lag;
        std::vector<int> randomVariablesTimesteps;

    public:
        DBNNodeTrainer(const Dataset& dataset, int node, int rvs, const VectorXd &weights);

        void train(double lambda, int maxIters) override;
    };
}


#endif //CDBAYES_DBNNODETRAINER_H

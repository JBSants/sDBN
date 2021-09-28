//
// Created by João Santos on 21/11/2020.
//

#ifndef CDBAYES_BNNODETRAINER_H
#define CDBAYES_BNNODETRAINER_H

#include "NodeTrainer.h"
#include "../generated/cdbayes_state_generated.h"

#include <Eigen/Dense>

using Eigen::VectorXd;
using Eigen::MatrixXd;

namespace RegularizedNetwork {
    class BNNodeTrainer : public NodeTrainer {
    protected:
        double stabilityFactor = 0.001;
        double alpha0 = 1;
        double eta = 0.5;
        double delta = 0.1;
        const Eigen::Ref<const VectorXd> weights;
        MatrixXd indicator;
        MatrixXd probabilities;
        int N;
        int M;

        void setBetasForParent(const MatrixXd &beta, int parent);
        inline Eigen::Block<MatrixXd> getBetasForParent(int parent) {
            return betas.block(0, dataset.randomVariablesIndices[parent],  betas.rows(), dataset.randomVariablesStates[parent]-1);
        }

        void setIntercepts(const Eigen::VectorXd &intercept);
        inline VectorXd getIntercepts() {
            return betas.col(betas.cols()-1);
        }

        void linearMapping(Eigen::Ref<MatrixXd> lm);
        void updateIndicator();

        double gradient(Eigen::Ref<MatrixXd> gradient, int parent, double lambda, const Eigen::Ref<const MatrixXd>, double groupCost);
        double interceptGradient(Eigen::Ref<VectorXd> gradient, double lambda, const Eigen::Ref<const MatrixXd> lm, double groupCost);
        double cost(int parent, double lambda, const Eigen::Ref<const MatrixXd>, double groupCost);
        double computeGroupCost(double lambda);

        double hessianCoefficient(int parent, const MatrixXd &lm);
        double interceptHessianCoefficient(const MatrixXd &lm);

    public:
        MatrixXd betas;

        BNNodeTrainer(const Dataset &dataset, int node, int rvs, const Eigen::Ref<const VectorXd> weights);

        std::vector<Eigen::SparseMatrix<double>> trainPath(const std::vector<double> &lambdas) override;
        void train(double lambda, int iters) override;
    };
}


#endif //CDBAYES_BNNODETRAINER_H

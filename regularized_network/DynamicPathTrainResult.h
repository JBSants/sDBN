//
// Created by João Santos on 28/12/2020.
//

#ifndef CDBAYES_DYNAMICPATHTRAINRESULT_H
#define CDBAYES_DYNAMICPATHTRAINRESULT_H

#include "PathTrainResult.h"
namespace RegularizedNetwork {
    class DynamicPathTrainResult : public PathTrainResult{
    private:
        std::vector<Eigen::SparseMatrix<double>> pathBetas;
        std::unordered_map<double, int> pathIndex;
        std::vector<double> path;
        int rvs;
        const Dataset &dataset;

    public:
        DynamicPathTrainResult(const Dataset &dataset) : rvs(dataset.randomVariablesStates.size()), dataset(dataset) {}

        void addLambda(double lambda) {
            if (!pathIndex.count(lambda)) {
                pathIndex[lambda] = path.size();
                path.emplace_back(lambda);

                for (int i = 0; i < rvs; i++) {
                    pathBetas.emplace_back(Eigen::SparseMatrix<double>(dataset.randomVariablesStates[i], dataset.dataset->cols()));
                }
            }
        }

        void setBetas(int rv, int idx, Eigen::SparseMatrix<double> &&betas) override {
            pathBetas[rv + idx * rvs] = betas;
        }

        void setBetas(int rv, double lambda, Eigen::SparseMatrix<double> &&betas) override {
            setBetas(rv, pathIndex.at(lambda), std::move(betas));
        }

        const Eigen::SparseMatrix<double> &getBetas(int rv, int idx) const override {
            return pathBetas[rv + idx * rvs];
        }

        const Eigen::SparseMatrix<double> &getBetas(int rv, double lambda) const override {
            return getBetas(rv, pathIndex.at(lambda));
        }

        virtual const std::vector<double> &getLambdas() const override {
            return path;
        }

        virtual const int getRVCount() const override {
            return rvs;
        }
    };
}

#endif //CDBAYES_DYNAMICPATHTRAINRESULT_H

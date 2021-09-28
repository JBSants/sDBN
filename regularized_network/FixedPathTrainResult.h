//
// Created by João Santos on 28/12/2020.
//

#ifndef CDBAYES_FIXEDPATHTRAINRESULT_H
#define CDBAYES_FIXEDPATHTRAINRESULT_H

#include "PathTrainResult.h"

namespace RegularizedNetwork {
    class FixedPathTrainResult : public PathTrainResult {
    private:
        std::vector<Eigen::SparseMatrix<double>> pathBetas;
        std::unordered_map<double, int> pathIndex;
        std::vector<double> path;
        int pathSize;

        void updatePathIndexAndSize() {
            pathSize = 0;
            for (double lambda : path) {
                pathIndex[lambda] = pathSize++;
            }
        }

    public:
        FixedPathTrainResult(const Dataset &dataset, const std::vector<double> &path) : path(path) {
            this->path = path;
            updatePathIndexAndSize();

            for (int i = 0; i < dataset.randomVariablesStates.size(); i++) {
                for (int j = 0; j < pathSize; j++) {
                    pathBetas.emplace_back(Eigen::SparseMatrix<double>(dataset.randomVariablesStates[i], dataset.dataset->cols()));
                }
            }
        }

        FixedPathTrainResult(std::vector<double> &&path, std::vector<Eigen::SparseMatrix<double>> &&pathBetas) {
            this->path = std::move(path);
            this->pathBetas = std::move(pathBetas);
            updatePathIndexAndSize();
        }

        FixedPathTrainResult(const PathTrainResult &result) {
            this->path = result.getLambdas();
            updatePathIndexAndSize();

            this->pathBetas.resize(pathSize*result.getRVCount());

            for (int i = 0; i < result.getRVCount(); i++) {
                for (int j = 0; j < path.size(); j++) {
                    setBetas(i, j, Eigen::SparseMatrix<double>(result.getBetas(i, j)));
                }
            }

        }

        void setBetas(int rv, int idx, Eigen::SparseMatrix<double> &&betas) override {
            pathBetas[idx + rv * pathSize] = betas;
        }

        void setBetas(int rv, double lambda, Eigen::SparseMatrix<double> &&betas) override {
            setBetas(rv, pathIndex.at(lambda), std::move(betas));
        }

        const Eigen::SparseMatrix<double> &getBetas(int rv, int idx) const override {
            return pathBetas[idx + rv * pathSize];
        }

        const Eigen::SparseMatrix<double> &getBetas(int rv, double lambda) const override {
            return getBetas(rv, pathIndex.at(lambda));
        }

        virtual const std::vector<double> &getLambdas() const override {
            return path;
        }

        virtual const int getRVCount() const override {
            return pathBetas.size() / pathSize;
        }
    };
}

#endif //CDBAYES_FIXEDPATHTRAINRESULT_H

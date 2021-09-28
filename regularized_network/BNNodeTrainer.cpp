//
// Created by João Santos on 21/11/2020.
//

#include "BNNodeTrainer.h"

#include <iostream>
#include "common/instrumentation.h"
#include "NodeTrainer.h"

RegularizedNetwork::BNNodeTrainer::BNNodeTrainer(const Dataset &dataset, int node, int rvs, const Eigen::Ref<const VectorXd> weights) : NodeTrainer(
        dataset, node, rvs), weights(weights) {
    N = dataset.dataset->cols();
    M = dataset.dataset->rows();
    betas = MatrixXd::Zero(dataset.randomVariablesStates[node], N);
    indicator = MatrixXd(M, dataset.randomVariablesStates[node]);
    probabilities = MatrixXd(M, dataset.randomVariablesStates[node]);
    updateIndicator();
}

void RegularizedNetwork::BNNodeTrainer::train(double lambda, int maxIters) {
    PROFILE_FUNCTION();
    const int rj = dataset.randomVariablesStates[node];
    const int maxRi = *std::max_element(dataset.randomVariablesStates.begin(), dataset.randomVariablesStates.end());
    MatrixXd lm(M, rj);
    MatrixXd linearMappingDiff(M, rj);
    std::vector<double> hVals;
    MatrixXd tempBeta(rj, maxRi-1);
    MatrixXd gradTemp(rj, maxRi-1);
    VectorXd interceptsGrad = VectorXd::Zero(rj);
    VectorXd newIntercepts(rj);
    double groupCost = computeGroupCost(lambda);

    linearMapping(lm);
    double interceptH = interceptHessianCoefficient(lm);

    int iter;
    for (iter = 0; iter < maxIters; iter++) {
        PROFILE_SCOPE("CD Iteration");
        double bestImprov = 0;
        double lastCost;

        for (int i = 0; i < V; i++) {
            PROFILE_SCOPE("Node Iteration");
            if (hVals.size() <= i) hVals.emplace_back(hessianCoefficient(i, lm));

            if (i == node) continue;

            const MatrixXd &oldBeta = getBetasForParent(i);
            Eigen::Ref<MatrixXd> grad = gradTemp.leftCols(dataset.randomVariablesStates[i] - 1);
            const double oldCost = gradient(grad, i, lambda, lm, groupCost);
            const double h = hVals[i];

            Eigen::Ref<MatrixXd> betaOptimal = tempBeta.leftCols(dataset.randomVariablesStates[i]-1);
            betaOptimal = grad - h * oldBeta;
            const double lbd = lambda * weights(i);
            const double djiNorm = betaOptimal.norm();
            if (djiNorm <= lbd) {
                betaOptimal.fill(0);
            } else {
                betaOptimal *= (-(1/h) * (1 - lbd / djiNorm));
            }

            MatrixXd direction = betaOptimal - oldBeta;
            double directionNorm = direction.norm();
            double oldBetaNorm = oldBeta.norm();

            if (directionNorm < 1e-4) continue;

            double deltaFactor = -grad.cwiseProduct(direction).sum() + lbd * (betaOptimal.norm() - oldBetaNorm);
            double alpha = alpha0;
            double improv, newCost;

            linearMappingDiff = alpha * dataset.dataset->block(0, dataset.randomVariablesIndices[i], M, dataset.randomVariablesStates[i] - 1) * direction.transpose();
            double groupCostDiff = lbd * ((oldBeta + alpha * direction).norm() - oldBetaNorm);

            {
                PROFILE_SCOPE("Line Search");
                auto start = std::chrono::high_resolution_clock::now();
                for (int j = 0; j < 15; j++) {
                    setBetasForParent(oldBeta + alpha * direction, i);
                    newCost = cost(i, lambda, lm + linearMappingDiff, groupCost + groupCostDiff);

                    if (newCost <= (oldCost + delta * alpha * deltaFactor)) {
                        break;
                    }

                    alpha *= eta;
                    linearMappingDiff *= eta;
                    groupCostDiff = lbd * ((oldBeta + alpha * direction).norm() - oldBetaNorm);
                }
                auto end = std::chrono::high_resolution_clock::now();

                improv = alpha * directionNorm;
                lm += linearMappingDiff;
                groupCost += groupCostDiff;

                if (improv > bestImprov) {
                    bestImprov = improv;
                }
                lastCost = newCost;

                interceptGradient(interceptsGrad, lambda, lm, groupCost);

                setIntercepts(getIntercepts() - interceptsGrad / interceptH);
                lm.rowwise() -= interceptsGrad.transpose() / interceptH;

                improv = interceptsGrad.norm() / abs(interceptH);
                if (improv > bestImprov) {
                    bestImprov = improv;
                }
            }
        }

        if (bestImprov < 1e-4) {
            break;
        }

        if ((iter % 10) == 0) {
            std::cerr << "[Node " << node << "] Iteration: " << iter << " (lik=" << lastCost << ")" << std::endl;
        }
    }
}

std::vector<Eigen::SparseMatrix<double>> RegularizedNetwork::BNNodeTrainer::trainPath(const std::vector<double> &lambdas) {
    std::vector<Eigen::SparseMatrix<double>> result;

    for (double lambda: lambdas) {
        train(lambda, 500);
        result.emplace_back(betas.sparseView());
        std::cerr << "[Node " << node << "] Lambda " << lambda << " complete" << std::endl;
    }

    return result;
}

void RegularizedNetwork::BNNodeTrainer::setBetasForParent(const MatrixXd &beta, const int parent) {
    const int l = dataset.randomVariablesIndices[parent];
    const int rj = betas.rows();
    const int ri = dataset.randomVariablesStates[parent];
    betas.block(0, l, rj, ri-1) = beta;
}


void RegularizedNetwork::BNNodeTrainer::setIntercepts(const Eigen::VectorXd &intercept) {
    betas.col(betas.cols()-1) = intercept;
}

void RegularizedNetwork::BNNodeTrainer::linearMapping(Eigen::Ref<MatrixXd> lm) {
    lm = *dataset.dataset * betas.transpose();
}

void RegularizedNetwork::BNNodeTrainer::updateIndicator() {
    const int rj = dataset.randomVariablesStates[node];
    const int l = dataset.randomVariablesIndices[node];
    for (int i = 0; i < M; i++) {
        indicator(i, 0) = 1;

        for (int j = 0; j < rj-1; j++) {
            if ((*dataset.dataset)(i, l + j) != 0) {
                indicator(i, j+1) = 1;
                indicator(i, 0) = 0;
            } else {
                indicator(i, j+1) = 0;
            }
        }
    }
}

double RegularizedNetwork::BNNodeTrainer::gradient(Eigen::Ref<MatrixXd> gradient, int parent, double lambda, const Eigen::Ref<const MatrixXd> lm, double groupCost) {
    PROFILE_FUNCTION();
    const int ri = dataset.randomVariablesStates[parent];
    const int rj = dataset.randomVariablesStates[node];
    const int l = dataset.randomVariablesIndices[parent];
    const int r = l + (ri - 1);
    double result = 0.0;

    for (int i = 0; i < M; i++) {
        for (int j = 0; j < rj; j++) {
            if (indicator(i, j) != 0.0) {
                result -= lm(i, j);
                break;
            }
        }
    }

    probabilities = lm.array().exp();

    for (int i = 0; i < M; i++) {
        double rowSum = probabilities.row(i).sum();
        result += log(rowSum);
        probabilities.row(i) = indicator.row(i) - probabilities.row(i) / rowSum;
    }

    result /= M;
    gradient = probabilities.transpose() * dataset.dataset->block(0, l, M, r-l);
    gradient /= M;

    /*for (int i = 0; i < V; i++) {
        result += lambda * weights(i) * getBetasForParent(i).norm();
    }*/

    result += groupCost;

    return result;
}

double RegularizedNetwork::BNNodeTrainer::cost(int parent, double lambda, const Eigen::Ref<const MatrixXd> lm, double groupCost) {
    double result = 0.0;
    int rj = dataset.randomVariablesStates[node];

    for (int i = 0; i < M; i++) {
        for (int j = 0; j < rj; j++) {
            if (indicator(i, j) != 0.0) {
                result -= lm(i, j);
                break;
            }
        }
    }

    probabilities = lm.array().exp();

    for (int i = 0; i < M; i++) {
        result += log(probabilities.row(i).sum());
    }

    result /= M;

    result += groupCost;

    /*for (int i = 0; i < V; i++) {
        result += lambda * weights(i) * getBetasForParent(i).norm();
    }*/

    return result;
}

double RegularizedNetwork::BNNodeTrainer::computeGroupCost(double lambda) {
    double result = 0.0;

    for (int i = 0; i < V; i++) {
        result += lambda * weights(i) * getBetasForParent(i).norm();
    }

    return result;
}

double RegularizedNetwork::BNNodeTrainer::interceptGradient(Eigen::Ref<VectorXd> gradient, double lambda, const Eigen::Ref<const MatrixXd> lm, double groupCost) {
    PROFILE_FUNCTION();
    const int rj = dataset.randomVariablesStates[node];
    double result = 0.0;

    for (int i = 0; i < M; i++) {
        for (int j = 0; j < rj; j++) {
            if (indicator(i, j) != 0.0) {
                result -= lm(i, j);
                break;
            }
        }
    }

    probabilities = lm.array().exp();

    for (int i = 0; i < rj; i++) {
        gradient(i) = 0.0;
    }

    for (int i = 0; i < M; i++) {
        double rowSum = probabilities.row(i).sum();
        result += log(rowSum);
        gradient += indicator.row(i) - probabilities.row(i) / rowSum;
    }

    gradient(0) = 0.0;
    gradient /= M;

    result /= M;

    /*for (int i = 0; i < V; i++) {
        result += lambda * weights(i) * getBetasForParent(i).norm();
    }*/

    result += groupCost;

    return result;
}

double RegularizedNetwork::BNNodeTrainer::hessianCoefficient(int parent, const MatrixXd &lm) {
    const int ri = dataset.randomVariablesStates[parent];
    const int rj = dataset.randomVariablesStates[node];
    MatrixXd result = MatrixXd::Zero(rj, ri-1);

    probabilities = lm.array().exp();

    for (int m = 0; m < M; m++) {
        probabilities.row(m) /= probabilities.row(m).sum();
        for (int j = 0; j < rj; j++) {
            for (int i = 0; i < ri-1; i++) {
                if ((*dataset.dataset)(m,dataset.randomVariablesIndices[parent]+i) != 0.0) {
                    result(j, i) += probabilities(m, j) * (1 - probabilities(m, j));
                }
            }
        }
    }

    return -std::max(result.maxCoeff(), stabilityFactor) / M;
}

double RegularizedNetwork::BNNodeTrainer::interceptHessianCoefficient(const MatrixXd &lm) {
    const int rj = dataset.randomVariablesStates[node];
    VectorXd result = VectorXd::Zero(rj);

    probabilities = lm.array().exp();

    for (int m = 0; m < M; m++) {
        probabilities.row(m) /= probabilities.row(m).sum();
        for (int j = 0; j < rj; j++) {
            result(j) += probabilities(m, j) * (1 - probabilities(m, j));
        }
    }

    return -std::max(result.maxCoeff(), stabilityFactor) / M;
}
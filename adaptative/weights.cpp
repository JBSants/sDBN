//
// Created by João Santos on 15/11/2020.
//

#include "weights.h"

#include "MultinomialLogisticRegression.h"
#include <queue>
#include <iostream>

#ifdef MPI_BUILD
    #include <mpi.h>

    #define BLOCK_LOW(id,p,n) ((id)*(n)/(p))
    #define BLOCK_HIGH(id,p,n) (BLOCK_LOW((id)+1,p,n)-1)
    #define BLOCK_SIZE(id,p,n) (BLOCK_HIGH(id,p,n)-BLOCK_LOW(id,p,n)+1)
#endif

using Eigen::MatrixXd;

void fullLevelIndicator(const MatrixXd &levelIndicator, MatrixXd &fullLevel) {
    for (int i = 0; i < levelIndicator.rows(); i++) {
        fullLevel(i,0) = 1;
        for (int j = 0; j < levelIndicator.cols(); j++) {
            if (levelIndicator(i, j) != 0.0) {
                fullLevel(i, j + 1) = 1;
                fullLevel(i, 0) = 0;
            } else {
                fullLevel(i,j+1) = 0;
            }
        }
    }
}

void findWeightsForRV(const int rv, const RegularizedNetwork::Dataset &d, const MultinomialLogisticRegression &regression, MatrixXd &coeff) {
    int rows = d.dataset->rows();
    int cols = d.dataset->cols();
    int states = d.randomVariablesStates[rv];
    int rvIndex = d.randomVariablesIndices[rv];
    MatrixXd y(rows, states);

    fullLevelIndicator(d.dataset->block(0, rvIndex, rows, states-1), y);

    MatrixXd data = MatrixXd(rows, cols-(states-1));
    data.block(0, 0, rows, rvIndex) = d.dataset->block(0,0, rows, rvIndex);
    data.block(0, rvIndex, rows, cols-(states-1)-rvIndex) = d.dataset->block(0,rvIndex+(states-1), rows, cols-(states-1)-rvIndex);

    regression.train(data, y, coeff);
}

void localDiscoverAdaptativeWeights(const RegularizedNetwork::Dataset &dataset, Eigen::MatrixXd &weights, const double gamma, const double epsilon) {
    const int rvs = dataset.randomVariablesStates.size();
    MultinomialLogisticRegression regression;
    regression.epsilon = epsilon;

    for (int i = 0; i < rvs; i++) {
        const int rj = dataset.randomVariablesStates[i];
        MatrixXd rvWeights(dataset.dataset->cols() - rj + 1, rj);
        findWeightsForRV(i, dataset, regression, rvWeights);

        for (int j = 0; j < rvs; j++) {
            if (j == i) {
                weights(i, j) = 0;
                continue;
            }

            int l = dataset.randomVariablesIndices[j];
            const int ri = dataset.randomVariablesStates[j];

            if (j > i) l -= rj - 1;

            weights(i, j) = 1 / std::pow(
                    std::max(rvWeights.block(l, 0, ri - 1, rj).norm(), 1e-4),
                    gamma);
        }
    }
}

#ifdef MPI_BUILD
void distributedDiscoverAdaptativeWeights(const RegularizedNetwork::Dataset &dataset, Eigen::MatrixXd &weights, const double gamma, const double epsilon) {
    const int rvs = dataset.randomVariablesStates.size();
    int rank, size;
    double *data = nullptr, *local;
    MultinomialLogisticRegression regression;
    regression.epsilon = epsilon;

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int low = BLOCK_LOW(rank, size, rvs);
    int high = BLOCK_HIGH(rank,size,rvs);
    int blockSize = BLOCK_SIZE(rank, size, rvs);

    std::cerr << "Doing from " << low << " to " << high << std::endl;

    local = new double [blockSize*rvs];
    Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> localResult(local, blockSize, rvs);
    for (int i = 0; i < blockSize; i++) {
        const int rv = low+i;
        const int rj = dataset.randomVariablesStates[rv];
        MatrixXd weights(dataset.dataset->cols() - rj + 1, rj);
        findWeightsForRV(rv, dataset, regression, weights);

        for (int j = 0; j < rvs; j++) {
            if (j == rv) {
                localResult(i, j) = 0;
                continue;
            }

            int l = dataset.randomVariablesIndices[j];
            const int ri = dataset.randomVariablesStates[j];

            if (j > rv) l -= rj - 1;

            localResult(i, j) = 1 / std::pow(
                    std::max(weights.block(l, 0, ri - 1, rj).norm(), 1e-4),
                    gamma);
        }
    }

    int displ[size], rcvCounts[size];

    data = new double[rvs*rvs];

    for (int i = 0; i < size; i++) {
        rcvCounts[i] = BLOCK_SIZE(i, size, rvs)*rvs;
        if (i > 0) {
            displ[i] = displ[i-1] + rcvCounts[i-1];
        } else {
            displ[i] = 0;
        }
        std::cerr << "cnts: " << rcvCounts[i] << "; displ: " << displ[i] << std::endl;
    }

    std::cerr << "Rank " << rank << " done." << "\n\n";

    MPI_Allgatherv(local, blockSize*rvs, MPI_DOUBLE, data, rcvCounts, displ, MPI_DOUBLE, MPI_COMM_WORLD);

    weights = Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(data, rvs, rvs);

    delete [] local;
    delete [] data;
}
#endif

void DiscoverAdaptativeWeights(const RegularizedNetwork::Dataset &dataset, Eigen::MatrixXd &weights, const double gamma, const double epsilon) {
#ifdef MPI_BUILD
    distributedDiscoverAdaptativeWeights(dataset, weights, gamma, epsilon);
#else
    localDiscoverAdaptativeWeights(dataset, weights, gamma, epsilon);
#endif
}

void updateMedian(double &median, double newValue, std::priority_queue<double> &s, std::priority_queue<double, std::vector<double>, std::greater<>> &g) {
    if (s.size() > g.size()) {
        if (newValue < median) {
            g.push(s.top());
            s.pop();
            s.push(newValue);
        } else {
            g.push(newValue);
        }

        median = (s.top() + g.top()) / 2.0;
    } else if (s.size() == g.size()) {
        if (newValue < median) {
            s.push(newValue);
            median = s.top();
        } else {
            g.push(newValue);
            median = g.top();
        }
    } else {
        if (newValue > median) {
            s.push(g.top());
            g.pop();
            g.push(newValue);
        } else {
            s.push(newValue);
        }

        median = (s.top() + g.top()) / 2.0;
    }
}

double FindMaxLambda(const RegularizedNetwork::Dataset &dataset, const Eigen::MatrixXd &weights) {
    double maxLambda = 0;
    const int rvs = dataset.randomVariablesStates.size();
    const int rows = dataset.dataset->rows();
    std::priority_queue<double> smaller;
    std::priority_queue<double, std::vector<double>, std::greater<>> greater;

    for (int i = 0; i < rvs; i++) {
        const int states = dataset.randomVariablesStates[i];
        MatrixXd y(rows, states);
        MatrixXd lm(rows, states);

        fullLevelIndicator(dataset.dataset->block(0, dataset.randomVariablesIndices[i], rows, states-1), y);

        for (int j = 0; j < rvs; j++) {
            if (i == j) continue;
            const int l = dataset.randomVariablesIndices[j];
            const int rj = dataset.randomVariablesStates[j];
            const int r = l + (rj - 1);

            for (int rw = 0; rw < rows; rw++) {
                lm.row(rw) = y.row(rw).array() - 1 / states;
            }

            double lambda = (lm.transpose() * dataset.dataset->block(0, l, rows, r-l) / rows).norm() / weights(i,j);

            if (lambda > maxLambda) maxLambda = lambda;
        }
    }

    return maxLambda;
}

//
// Created by João Santos on 29/11/2020.
//

#include "DBNNodeTrainer.h"
#include "BNNodeTrainer.h"
#include <iostream>

RegularizedNetwork::DBNNodeTrainer::DBNNodeTrainer(const Dataset &dataset, int node, int rvs, const VectorXd &weights) :
        RegularizedNetwork::BNNodeTrainer(dataset, node, rvs, weights) {
    auto timeseriesDataset = dynamic_cast<const TimeseriesDataset &>(dataset);
    timesteps = timeseriesDataset.timesteps;
    lag = timeseriesDataset.lag;
    randomVariablesTimesteps = timeseriesDataset.randomVariablesTimesteps;
}

void RegularizedNetwork::DBNNodeTrainer::train(double lambda, int maxIters) {
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
        double bestImprov = 0;
        double lastCost;

        for (int i = 0; i < V; i++) {
            if (hVals.size() <= i) hVals.emplace_back(hessianCoefficient(i, lm));

            if (i == node) continue;
            if (randomVariablesTimesteps[i] > randomVariablesTimesteps[node]) continue;
            if (randomVariablesTimesteps[node] >= 0 && randomVariablesTimesteps[node] - randomVariablesTimesteps[i] > lag) continue;

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

        if (bestImprov < 1e-4) {
            break;
        }

        if ((iter % 10) == 0) {
            std::cerr << "[Node " << node << "] Iteration: " << iter << " (lik=" << lastCost << ")" << std::endl;
        }
    }
}



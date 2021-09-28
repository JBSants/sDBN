//
// Created by João Santos on 28/12/2020.
//

#include "DBNPathTrainer.h"
#include "Worker.h"
#include "../DBNNodeTrainer.h"
#include "WorkPool.h"

void RegularizedNetwork::Distributed::DBNPathTrainer::spawnWorker() {
    Worker<DBNNodeTrainer>(dataset, weights, eraIterations).start();
}

void RegularizedNetwork::Distributed::DBNPathTrainer::managePool(PathTrainResult &result, int eras, int size,
                                                                const std::vector<double> &lambdas) {
    WorkPool workPool(lambdas, result);
    std::queue<int> initialPool;
    auto &timeseriesDataset = dynamic_cast<const TimeseriesDataset &>(dataset);
    const int rvsPerTimestep = (rvs - timeseriesDataset.staticRandomVariables) / timeseriesDataset.timesteps;

    if (!noState) {
        workPool.setOnPartialResult([this, lambdas](WorkPool::PartialResult partialResult) {
            updateStateBetas(partialResult.rv, partialResult.pathIndex, partialResult.betas, lambdas);
            writeState();
        });
    }

    if (noState || eras > 1) {
        for (int i = timeseriesDataset.stubTimesteps*rvsPerTimestep; i < rvs; i++) {
            initialPool.push(i);
        }
    }

    if (noState) {
        workPool.start(initialPool);
    } else {
        std::queue<int> pool;
        recoverState(pool, result, lambdas);
        workPool.start(pool);
    }

    for (int era = 0; era < eras; era++) {
        while (workPool.next());

        if (!noState) {
            updateStateEra();
            writeState();
        }

        if (era+1 < eras) {
            workPool.start(initialPool);
        }
    }
}
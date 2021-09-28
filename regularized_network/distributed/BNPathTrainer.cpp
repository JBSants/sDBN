//
// Created by João Santos on 21/11/2020.
//

#include "BNPathTrainer.h"
#include <queue>
#include <iostream>
#include <fstream>
#include "WorkPool.h"
#include "../BNNodeTrainer.h"
#include "../FixedPathTrainResult.h"

RegularizedNetwork::Distributed::BNPathTrainer::BNPathTrainer(const Dataset &dataset, const MatrixXd &weights, const std::string &state) :
        dataset(dataset),
        weights(weights),
        stateFilename(state) {
    rvs = dataset.randomVariablesStates.size();
    noState = stateFilename.empty();
}

void RegularizedNetwork::Distributed::BNPathTrainer::managePool(PathTrainResult &result, int eras, int size,
                                                                const std::vector<double> &lambdas) {
    WorkPool workPool(lambdas, result);

    if (!noState) {
        workPool.setOnPartialResult([this, lambdas](WorkPool::PartialResult partialResult) {
            updateStateBetas(partialResult.rv, partialResult.pathIndex, partialResult.betas, lambdas);
            writeState();
        });
    }

    if (noState) {
        workPool.start(rvs);
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
            workPool.start(rvs);
        }
    }
}

void RegularizedNetwork::Distributed::BNPathTrainer::recoverState(std::queue<int> &pool, PathTrainResult &result,
                                                                  const std::vector<double> &lambdas) {
    std::ifstream stateFile;
    stateFile.open(stateFilename, std::ifstream::binary);

    if (stateFile.is_open()) {
        std::istreambuf_iterator<char> start(stateFile), end;
        std::vector<char> buffer(start, end);
        auto verifier = flatbuffers::Verifier((uint8_t *)buffer.data(), buffer.size());
        if (VerifyCDBayesStateBuffer(verifier)) {
            GetCDBayesState(buffer.data())->UnPackTo(&state);
        } else {
            clearState(lambdas);
            writeState();
        }

        stateFile.close();
    } else {
        clearState(lambdas);
        writeState();
    }

    printState();

    for (int i = 0; i < rvs; i++) {
        bool found = false;
        for (int rv : state.currentEraRVs) {
            if (rv == i) {
                found = true;
                break;
            }
        }

        if (!found) pool.push(i);
    }

    for (int l = 0; l < state.lambdas.size(); l++) {
        for (int i = 0; i < rvs; i++) {
            MatrixXd beta = MatrixXd::Zero(dataset.randomVariablesStates[i], dataset.dataset->cols());

            for (const Triplet triplet : state.states[i]->path[l]->betas) {
                beta(triplet.i(), triplet.j()) = triplet.v();
            }

            result.setBetas(i, l, beta.sparseView());
        }
    }
}

void RegularizedNetwork::Distributed::BNPathTrainer::printState() {
    std::cerr << "CDBayesState(eras=" << state.eras
                                      << ", #lambdas=" << state.lambdas.size()
                                      << ", #currentEraRVs=" << state.currentEraRVs.size()
                                      << ")" << std::endl;
}

std::vector<Triplet> RegularizedNetwork::Distributed::BNPathTrainer::packBetas(const Eigen::SparseMatrix<double> &beta) {
    std::vector<Triplet> result;
    for (int k=0; k < beta.outerSize(); k++) {
        for (Eigen::SparseMatrix<double>::InnerIterator it(beta,k); it; ++it) {
            result.emplace_back(Triplet(it.row(), it.col(), it.value()));
        }
    }

    return result;
}

void RegularizedNetwork::Distributed::BNPathTrainer::updateStateBetas(int rv, int updatedPathIdx,
                                                                      const Eigen::SparseMatrix<double> &betas,
                                                                      const std::vector<double> &lambdas) {
    state.states[rv]->path[updatedPathIdx]->betas = packBetas(betas);
    state.states[rv]->lastPathIdx = ++updatedPathIdx;
    if (updatedPathIdx >= lambdas.size()) {
        state.states[rv]->lastPathIdx = 0;
        state.currentEraRVs.emplace_back(rv);
    }
}

void RegularizedNetwork::Distributed::BNPathTrainer::updateStateEra() {
    state.currentEraRVs.clear();
    state.eras += 1;
}

void RegularizedNetwork::Distributed::BNPathTrainer::writeState() {
    std::ofstream stateFile;
    stateFile.open(stateFilename, std::ofstream::binary);

    if (stateFile.is_open()) {
        flatbuffers::FlatBufferBuilder builder;

        builder.Finish(CDBayesState::Pack(builder, &state));
        stateFile.write((char *) builder.GetBufferPointer(), builder.GetSize());

        stateFile.close();
    }
}

void RegularizedNetwork::Distributed::BNPathTrainer::clearState(const std::vector<double> &lambdas) {
    state.lambdas = lambdas;
    state.eras = 0;
    state.states.clear();
    state.currentEraRVs.clear();

    for (int i = 0; i < rvs; i++) {
        state.states.emplace_back(std::make_unique<RegularizationPathStateT>());
        state.states[i]->rv = i;
        state.states[i]->lastPathIdx = 0;
        state.states[i]->done = false;

        for (int j = 0; j < lambdas.size(); j++) {
            state.states[i]->path.emplace_back(std::make_unique<RegularizationPathT>());
            state.states[i]->path[j]->lambda = lambdas[j];
            state.states[i]->path[j]->cost = INFINITY;
        }
    }
}

std::unique_ptr<RegularizedNetwork::PathTrainResult> RegularizedNetwork::Distributed::BNPathTrainer::train(const std::vector<double> &lambdas, int eras) {
    int rank, size;
    std::unique_ptr<PathTrainResult> trainResult;

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (rank == 0) {
        trainResult = std::make_unique<FixedPathTrainResult>(dataset, lambdas);

        managePool(*trainResult, eras, size, lambdas);
    } else {
        spawnWorker();
    }

    std::cerr << "[Rank " << rank << "] On barrier" << std::endl;

    MPI_Barrier(MPI_COMM_WORLD);

    std::cerr << "[Rank " << rank << "] After barrier" << std::endl;

    return trainResult;
}

void RegularizedNetwork::Distributed::BNPathTrainer::spawnWorker() {
    Worker<RegularizedNetwork::BNNodeTrainer>(dataset, weights, eraIterations).start();
}

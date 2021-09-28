//
// Created by João Santos on 26/12/2020.
//

#ifndef CDBAYES_REGULARIZEDGRAPHTRAINERWORKPOOL_H
#define CDBAYES_REGULARIZEDGRAPHTRAINERWORKPOOL_H

#include <Eigen/Sparse>
#include <memory>
#include <queue>
#include <mpi.h>
#include "WorkerRequest.h"
#include "WorkerResponse.h"
#include "../PathTrainResult.h"

namespace RegularizedNetwork {
    namespace Distributed {
        class WorkPool {
        public:
            struct PartialResult {
            public:
                const Eigen::SparseMatrix<double> &betas;
                int rv;
                int pathIndex;

                PartialResult(const Eigen::SparseMatrix<double> &betas, int rv, int pathIndex) : betas(betas), rv(rv), pathIndex(pathIndex) {};
            };

            struct Result {
                int rv;

                Result(int rv) : rv(rv) {};
                Result() : rv(-1) {};

                explicit operator bool() const {
                    return rv >= 0;
                }
            };
        private:
            struct WorkPoolRequest : WorkerRequest {
                int pathIndex;
                WorkPoolRequest(WorkerRequest::RequestType type, int rv, double lambda, int pathIndex) : WorkerRequest(type, rv, lambda), pathIndex(pathIndex) {};
            };

            std::queue<int> pool;
            const std::vector<double> &path;

            std::function<void(PartialResult)> onPartialResult;
            std::vector<WorkPoolRequest> requests;
            std::vector<WorkerResponse> responses;

            std::vector<MPI_Request> mpiRequests;

            PathTrainResult &trainResult;

            int rvs;
            int workersCapacity;

            void sendSparseMatrix(int worker, const Eigen::SparseMatrix<double> &matrix);
            void receiveSparseMatrix(int worker, Eigen::SparseMatrix<double> &matrix);
            void requestNext(int worker);
            void sendRequest(int worker);
            void terminateWorker(int worker);
        public:
            WorkPool(const std::vector<double> &path, PathTrainResult &trainResult);
            ~WorkPool();

            void start(int rvs);
            void start(const std::queue<int> &rvs);

            Result next();
            void setOnPartialResult(const std::function<void(PartialResult)> &callback);
        };
    }
}

#endif //CDBAYES_REGULARIZEDGRAPHTRAINERWORKPOOL_H

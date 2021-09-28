//
// Created by João Santos on 26/12/2020.
//

#include "WorkPool.h"
#include <mpi.h>

namespace RegularizedNetwork {
    namespace Distributed {
        WorkPool::WorkPool(const std::vector<double> &path, PathTrainResult &trainResult) : path(path), trainResult(trainResult) {
            int rank;

            MPI_Comm_rank(MPI_COMM_WORLD, &rank);

            if (rank != 0) exit(EXIT_FAILURE);

            MPI_Comm_size(MPI_COMM_WORLD, &workersCapacity);
            workersCapacity -= 1;

            requests.reserve(workersCapacity);
            mpiRequests.reserve(workersCapacity);
            responses.reserve(workersCapacity);
        }

        WorkPool::~WorkPool() {
            for (int i = 0; i < workersCapacity; i++) {
                terminateWorker(i);
            }
        }

        void WorkPool::terminateWorker(int worker) {
            WorkerRequest terminateRequest;

            MPI_Send(&terminateRequest, sizeof(WorkerRequest), MPI_BYTE, worker+1, 0, MPI_COMM_WORLD);
        }

        void WorkPool::sendSparseMatrix(int worker, const Eigen::SparseMatrix<double> &matrix) {
            assert(matrix.isCompressed());
            int nnz = matrix.nonZeros();
	    const int targetRank = worker+1;

            MPI_Send(&nnz, 1, MPI_INT, targetRank, 0, MPI_COMM_WORLD);
            MPI_Send(matrix.valuePtr(), nnz, MPI_DOUBLE, targetRank, 0, MPI_COMM_WORLD);
            MPI_Send(matrix.innerIndexPtr(), nnz, MPI_INT, targetRank, 0, MPI_COMM_WORLD);
            MPI_Send(matrix.outerIndexPtr(), matrix.cols(), MPI_INT, targetRank, 0, MPI_COMM_WORLD);
        }

        void WorkPool::receiveSparseMatrix(int worker, Eigen::SparseMatrix<double> &matrix) {
            const int cols = matrix.cols();
            const int targetRank = worker+1;
            int nnz;

            MPI_Recv(&nnz, 1, MPI_INT, targetRank, 0, MPI_COMM_WORLD, nullptr);

            matrix.reserve(nnz);
            matrix.resizeNonZeros(nnz);

            MPI_Recv(matrix.valuePtr(), nnz, MPI_DOUBLE, targetRank, 0, MPI_COMM_WORLD, nullptr);
            MPI_Recv(matrix.innerIndexPtr(), nnz, MPI_INT, targetRank, 0, MPI_COMM_WORLD, nullptr);
            MPI_Recv(matrix.outerIndexPtr(), cols, MPI_INT, targetRank, 0, MPI_COMM_WORLD, nullptr);
            matrix.outerIndexPtr()[cols] = nnz;
        }

        void WorkPool::sendRequest(int worker) {
            WorkPoolRequest &request = requests[worker];
            MPI_Send(static_cast<WorkerRequest *>(&request), sizeof(WorkerRequest), MPI_BYTE, worker+1, 0, MPI_COMM_WORLD);
            sendSparseMatrix(worker, trainResult.getBetas(request.rv, request.lambda));
            MPI_Irecv(&responses[worker], sizeof (WorkerResponse), MPI_BYTE, worker+1, 0, MPI_COMM_WORLD, &mpiRequests[worker]);
        }

        void WorkPool::requestNext(int worker) {
            if (requests.size() <= worker) {
                int rv = pool.front();

                mpiRequests.push_back(MPI_REQUEST_NULL);
                requests.emplace_back(WorkPoolRequest(WorkerRequest::NEW_LAMBDA, rv, path[0], 0));
                responses.push_back(-1);

                pool.pop();
            } else {
                WorkPoolRequest &request = requests[worker];
                request.pathIndex += 1;

                if (request.pathIndex >= path.size()) {
                    request.rv = pool.front();
                    request.pathIndex = 0;
                    pool.pop();
                }

                request.lambda = path[request.pathIndex];
            }

            sendRequest(worker);
        }

        void WorkPool::start(int newRvs) {
            rvs = newRvs;
            pool = std::queue<int>();

            for (int i = 0; i < newRvs; i++) pool.push(i);

            for (int i = 0; i < workersCapacity; i++) {
                if (!pool.empty()) {
                    requestNext(i);
                }
            }
        }

        void WorkPool::start(const std::queue<int> &newPool) {
            rvs = newPool.size();
            pool = newPool;

            for (int i = 0; i < workersCapacity; i++) {
                if (!pool.empty()) {
                    requestNext(i);
                }
            }
        }

        WorkPool::Result WorkPool::next() {
            Result result;
            int completedWorker;
            MPI_Waitany(mpiRequests.size(), mpiRequests.data(), &completedWorker, MPI_STATUS_IGNORE);

            if (completedWorker == MPI_UNDEFINED) return result;

            mpiRequests[completedWorker] = MPI_REQUEST_NULL;
            const WorkPoolRequest &completedRequest = requests[completedWorker];

            Eigen::SparseMatrix<double> betas(trainResult.getBetas(completedRequest.rv, completedRequest.pathIndex).rows(), trainResult.getBetas(completedRequest.rv, completedRequest.pathIndex).cols());

            receiveSparseMatrix(completedWorker, betas);

            trainResult.setBetas(completedRequest.rv, completedRequest.pathIndex, std::move(betas));

            if (onPartialResult) onPartialResult(PartialResult{trainResult.getBetas(completedRequest.rv, completedRequest.lambda), completedRequest.rv, completedRequest.pathIndex});

            result.rv = completedRequest.rv;

            if (!pool.empty()) requestNext(completedWorker);

            return result;
        }

        void WorkPool::setOnPartialResult(const std::function<void(PartialResult)> &callback) {
            onPartialResult = callback;
        }
    }
}

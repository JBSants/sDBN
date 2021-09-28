//
// Created by João Santos on 26/12/2020.
//

#ifndef CDBAYES_REGULARIZEDGRAPHTRAINERWORKER_H
#define CDBAYES_REGULARIZEDGRAPHTRAINERWORKER_H

#include <Eigen/Dense>
#include <mpi.h>
#include "regularized_network/dataset.h"
#include "WorkerRequest.h"
#include "WorkerResponse.h"

namespace RegularizedNetwork {
    namespace Distributed {
        template <typename T> class Worker {
            const Dataset &dataset;
            const Eigen::MatrixXd &weights;
            const int eraIterations;

            void receiveSparseMatrix(Eigen::SparseMatrix<double> &matrix) {
                const int cols = matrix.cols();
                int nnz;

                MPI_Recv(&nnz, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, nullptr);

                matrix.reserve(nnz);
                matrix.resizeNonZeros(nnz);

                MPI_Recv(matrix.valuePtr(), nnz, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, nullptr);
                MPI_Recv(matrix.innerIndexPtr(), nnz, MPI_INT, 0, 0, MPI_COMM_WORLD, nullptr);
                MPI_Recv(matrix.outerIndexPtr(), cols, MPI_INT, 0, 0, MPI_COMM_WORLD, nullptr);
                matrix.outerIndexPtr()[cols] = nnz;
            };

            void sendSparseMatrix(const Eigen::SparseMatrix<double> &matrix) {
                assert(matrix.isCompressed());
                int nnz = matrix.nonZeros();

                MPI_Send(&nnz, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
                MPI_Send(matrix.valuePtr(), nnz, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
                MPI_Send(matrix.innerIndexPtr(), nnz, MPI_INT, 0, 0, MPI_COMM_WORLD);
                MPI_Send(matrix.outerIndexPtr(), matrix.cols(), MPI_INT, 0, 0, MPI_COMM_WORLD);
            };
        public:
            Worker(const Dataset &dataset, const Eigen::MatrixXd &weights, int eraIterations = 5) : dataset(dataset), weights(weights), eraIterations(eraIterations) {};

            void start() {
                int rank;
                WorkerRequest request;

                MPI_Comm_rank(MPI_COMM_WORLD, &rank);

                while (true) {
                    MPI_Recv(&request, sizeof(WorkerRequest), MPI_BYTE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

                    if (request.type == WorkerRequest::TERMINATE) break;

                    Eigen::SparseMatrix<double> betas(dataset.randomVariablesStates[request.rv],
                                                      dataset.dataset->cols());

                    receiveSparseMatrix(betas);

                    Eigen::VectorXd weightsRow = weights.row(request.rv);

                    T trainer(dataset, request.rv, dataset.randomVariablesStates.size(), weightsRow);
                    trainer.betas = betas;

                    trainer.train(request.lambda, eraIterations);

                    WorkerResponse response = 0;
                    MPI_Send(&response, sizeof(WorkerResponse), MPI_BYTE, 0, 0, MPI_COMM_WORLD);
                    sendSparseMatrix(trainer.betas.sparseView());

                    std::cerr << "[Rank " << rank << "] Request " << request << " served." << std::endl;
                }
            };
        };
    }
}


#endif //CDBAYES_REGULARIZEDGRAPHTRAINERWORKER_H

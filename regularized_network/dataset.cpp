//
// Created by João Santos on 12/11/2020.
//

#include "dataset.h"

#include <iostream>
#include <openssl/md5.h>

#define CHECKSUM_SIZE 16

namespace RegularizedNetwork {

    void readMatrix(std::istream &stream, Eigen::Ref<Eigen::MatrixXd> matrix, int m, int n, MD5_CTX &ctx) {
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                stream >> matrix(i, j);
                MD5_Update(&ctx, &matrix(i, j), sizeof(double));
            }
        }
    }

    void readVector(std::istream &stream, std::vector<int> &vec, MD5_CTX &ctx) {
        for (int &el : vec) {
            stream >> el;
            MD5_Update(&ctx, &el, sizeof(int));
        }
    }

    void indicesFromStates(const std::vector<int> &states, std::vector<int> &indices) {
        indices[0] = 0;
        for (int i = 1; i < indices.size(); i++) {
            indices[i] = indices[i - 1] + states[i - 1] - 1;
        }
    }

    std::string checksumHexString(unsigned char *checksum) {
        std::stringstream result;

        for (int i = 0; i < CHECKSUM_SIZE; i++) {
            result << std::hex << (int) checksum[i];
        }

        return result.str();
    }

    Dataset ReadDataset(std::istream &stream) {
        MD5_CTX ctx;
        unsigned char checksum[CHECKSUM_SIZE];
        int rvs, m, cols;

        stream >> rvs >> m >> cols;

        MD5_Init(&ctx);

        MD5_Update(&ctx, &rvs, sizeof(int));
        MD5_Update(&ctx, &m, sizeof(int));
        MD5_Update(&ctx, &cols, sizeof(int));

        auto *datasetMatrix = new Eigen::MatrixXd(m, cols + 1);
        readMatrix(stream, *datasetMatrix, m, cols, ctx);
        datasetMatrix->col(cols) = Eigen::MatrixXd::Ones(m, 1);

        std::vector<int> states(rvs);
        std::vector<int> indices(rvs);

        readVector(stream, states, ctx);
        indicesFromStates(states, indices);

        MD5_Final(checksum, &ctx);

        return Dataset(datasetMatrix, states, indices, checksumHexString(checksum));
    }

    struct TimeseriesDatasetHeader {
        int staticRandomVariables;
        int rvsPerTimestep;
        int m;
        int cols;
        int timesteps;
    };

    TimeseriesDatasetHeader ReadTimeseriesRawDataset(std::istream &stream, Eigen::MatrixXd *datasetMatrix, std::vector<int> &states, std::vector<int> &timestepsVector, MD5_CTX &ctx) {
        TimeseriesDatasetHeader header = { 0 };

        stream >> header.staticRandomVariables >> header.rvsPerTimestep >> header.timesteps >> header.m >> header.cols;

        MD5_Init(&ctx);

        MD5_Update(&ctx, &header.staticRandomVariables, sizeof(int));
        MD5_Update(&ctx, &header.rvsPerTimestep, sizeof(int));
        MD5_Update(&ctx, &header.timesteps, sizeof(int));
        MD5_Update(&ctx, &header.m, sizeof(int));
        MD5_Update(&ctx, &header.cols, sizeof(int));

        datasetMatrix->resize(header.m, header.cols+1);
        readMatrix(stream, *datasetMatrix, header.m, header.cols, ctx);
        datasetMatrix->col(header.cols) = Eigen::MatrixXd::Ones(header.m, 1);

        int nonStaticRvs = header.rvsPerTimestep*header.timesteps;
        int rvs = nonStaticRvs + header.staticRandomVariables;

        states.resize(header.rvsPerTimestep+header.staticRandomVariables);
        timestepsVector.resize(rvs);

        readVector(stream, states, ctx);
        readVector(stream, timestepsVector, ctx);

        return header;
    }


    TimeseriesDataset ReadStationaryTimeseriesDataset(std::istream &stream, int lag) {
        MD5_CTX ctx;
        unsigned char checksum[16];
        const bool stationary = true;

        Eigen::MatrixXd datasetMatrix;
        std::vector<int> states;
        std::vector<int> timestepsVector;

        auto header = ReadTimeseriesRawDataset(stream, &datasetMatrix, states, timestepsVector, ctx);

        MD5_Update(&ctx, &lag, sizeof(int));
        MD5_Update(&ctx, &stationary, sizeof(bool));
        MD5_Final(checksum, &ctx);

        int staticRandomVariablesCols = 0;
        for (int i = 0; i < header.staticRandomVariables; i++) { // Find how many static random variables exist
            staticRandomVariablesCols += states[i] - 1;
        }

        int colsPerTimestep = (header.cols - staticRandomVariablesCols) / header.timesteps;

        if (lag <= 0) lag = header.timesteps - 1;
        int transitionSlices =  lag + 1;
        int stubSlices = lag;
        int rolls = header.timesteps - transitionSlices + 1;
        auto transitionDatasetMatrix = new Eigen::MatrixXd(rolls * header.m, staticRandomVariablesCols +
                                                                             transitionSlices * colsPerTimestep + 1);
        std::vector<int> transitionStates(header.staticRandomVariables + transitionSlices * header.rvsPerTimestep);
        std::vector<int> transitionIndices(header.staticRandomVariables + transitionSlices * header.rvsPerTimestep);
        std::vector<int> transitionTimesteps(header.staticRandomVariables + transitionSlices * header.rvsPerTimestep);

        for (int i = 0; i < rolls; i++) {
            transitionDatasetMatrix->block(i * header.m, 0, header.m, staticRandomVariablesCols) = datasetMatrix.block(0, 0, header.m,
                                                                                                                       staticRandomVariablesCols);
            transitionDatasetMatrix->block(i * header.m, staticRandomVariablesCols, header.m,
                                           transitionSlices * colsPerTimestep) = datasetMatrix.block(0,
                                                                                                     staticRandomVariablesCols +
                                                                                                     i *
                                                                                                     colsPerTimestep, header.m,
                                                                                                     transitionSlices *
                                                                                                     colsPerTimestep);
        }
        transitionDatasetMatrix->col(transitionDatasetMatrix->cols() - 1) = Eigen::MatrixXd::Ones(rolls * header.m, 1);

        int lastCol = 0;

        for (int i = 0; i < header.staticRandomVariables; i++) {
            transitionStates[i] = states[i];
            transitionIndices[i] = lastCol;
            transitionTimesteps[i] = -1;

            lastCol += states[i] - 1;
        }

        for (int i = 0; i < transitionSlices; i++) {
            for (int j = 0; j < header.rvsPerTimestep; j++) {
                transitionStates[header.staticRandomVariables + j + i * header.rvsPerTimestep] = states[header.staticRandomVariables +
                                                                                                        j];
                transitionIndices[header.staticRandomVariables + j + i * header.rvsPerTimestep] = lastCol;
                transitionTimesteps[header.staticRandomVariables + j + i * header.rvsPerTimestep] = i;

                lastCol += states[header.staticRandomVariables + j] - 1;
            }
        }

        return TimeseriesDataset(transitionDatasetMatrix, transitionStates, transitionIndices, transitionTimesteps,
                                 transitionSlices, stubSlices, header.staticRandomVariables, lag, checksumHexString(checksum));
    }

    TimeseriesDataset ReadRolledStationaryTimeseriesDataset(std::istream &stream, int lag) {
        MD5_CTX ctx;
        unsigned char checksum[16];
        const bool stationary = true;

        auto datasetMatrix = new Eigen::MatrixXd();
        std::vector<int> states;
        std::vector<int> timestepsVector;

        auto header = ReadTimeseriesRawDataset(stream, datasetMatrix, states, timestepsVector, ctx);

        if (lag <= 0) lag = header.timesteps - 1;

        MD5_Update(&ctx, &lag, sizeof(int));
        MD5_Update(&ctx, &stationary, sizeof(bool));
        MD5_Final(checksum, &ctx);

        int lastCol = 0;
        std::vector<int> indices(header.staticRandomVariables + header.timesteps * header.rvsPerTimestep);
        std::vector<int> transitionStates(header.staticRandomVariables + header.timesteps * header.rvsPerTimestep);

        for (int i = 0; i < header.staticRandomVariables; i++) {
            indices[i] = lastCol;
            lastCol += states[i] - 1;
        }

        for (int i = 0; i < header.timesteps; i++) {
            for (int j = 0; j < header.rvsPerTimestep; j++) {
                indices[header.staticRandomVariables + j + i * header.rvsPerTimestep] = lastCol;
                transitionStates[header.staticRandomVariables + j + i * header.rvsPerTimestep] = states[header.staticRandomVariables +
                                                                                                        j];
                lastCol += states[header.staticRandomVariables + j] - 1;
            }
        }

        return TimeseriesDataset(datasetMatrix, transitionStates, indices, timestepsVector,
                                 header.timesteps, lag, header.staticRandomVariables, lag, checksumHexString(checksum));
    }


    TimeseriesDataset ReadTimeseriesDataset(std::istream &stream, int lag) {
        MD5_CTX ctx;
        unsigned char checksum[16];

        auto datasetMatrix = new Eigen::MatrixXd();
        std::vector<int> states;
        std::vector<int> timestepsVector;

        auto header = ReadTimeseriesRawDataset(stream, datasetMatrix, states, timestepsVector, ctx);

        if (lag <= 0) lag = header.timesteps - 1;

        MD5_Update(&ctx, &lag, sizeof(int));
        MD5_Final(checksum, &ctx);

        int lastCol = 0;
        std::vector<int> indices(header.staticRandomVariables + header.timesteps * header.rvsPerTimestep);
        std::vector<int> transitionStates(header.staticRandomVariables + header.timesteps * header.rvsPerTimestep);

        for (int i = 0; i < header.staticRandomVariables; i++) {
            indices[i] = lastCol;
            lastCol += states[i] - 1;
        }

        for (int i = 0; i < header.timesteps; i++) {
            for (int j = 0; j < header.rvsPerTimestep; j++) {
                indices[header.staticRandomVariables + j + i * header.rvsPerTimestep] = lastCol;
                transitionStates[header.staticRandomVariables + j + i * header.rvsPerTimestep] = states[header.staticRandomVariables +
                                                                                                        j];
                lastCol += states[header.staticRandomVariables + j] - 1;
            }
        }

        return TimeseriesDataset(datasetMatrix, transitionStates, indices, timestepsVector,
                                  header.timesteps, 1, header.staticRandomVariables, lag, checksumHexString(checksum));
    }
}
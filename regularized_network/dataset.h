//
// Created by João Santos on 12/11/2020.
//

#ifndef CDBAYES_DATASET_READER_H
#define CDBAYES_DATASET_READER_H

#include <Eigen/Eigen>
#include <istream>
#include <memory>

namespace RegularizedNetwork {
    class Dataset {
    public:
        std::shared_ptr<const Eigen::MatrixXd> dataset;
        std::vector<int> randomVariablesStates;
        std::vector<int> randomVariablesIndices;
        std::string checksum;

        Dataset(Eigen::MatrixXd *dataset,
                std::vector<int> randomVariablesStates,
                std::vector<int> randomVariablesIndices,
                std::string checksum) :
                dataset(std::shared_ptr<const Eigen::MatrixXd>(dataset)),
                randomVariablesStates(std::move(randomVariablesStates)),
                randomVariablesIndices(std::move(randomVariablesIndices)),
                checksum(checksum) {};

        Dataset() = default;

        virtual ~Dataset() {};
    };

    class TimeseriesDataset : public Dataset {
    public:
        int timesteps = 0;
        int stubTimesteps = 0;
        int staticRandomVariables = 0;
        int lag = 0;
        std::vector<int> randomVariablesTimesteps;

        TimeseriesDataset(Eigen::MatrixXd *dataset,
                          std::vector<int> randomVariablesStates,
                          std::vector<int> randomVariablesIndices,
                          std::vector<int> randomVariablesTimesteps,
                          int timesteps,
                          int stubTimesteps,
                          int staticRandomVariables,
                          int lag,
                          std::string checksum) :
                Dataset(dataset, std::move(randomVariablesStates), std::move(randomVariablesIndices),
                        std::move(checksum)),
                timesteps(timesteps),
                stubTimesteps(stubTimesteps),
                staticRandomVariables(staticRandomVariables),
                lag(lag),
                randomVariablesTimesteps(std::move(randomVariablesTimesteps)) {};

        TimeseriesDataset() = default;
    };

    Dataset ReadDataset(std::istream &stream);
    TimeseriesDataset ReadTimeseriesDataset(std::istream &stream, int lag);
    TimeseriesDataset ReadRolledStationaryTimeseriesDataset(std::istream &stream, int lag);
    TimeseriesDataset ReadStationaryTimeseriesDataset(std::istream &stream, int lag);
}

#endif //CDBAYES_DATASET_READER_H

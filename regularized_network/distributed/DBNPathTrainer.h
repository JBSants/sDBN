//
// Created by João Santos on 28/12/2020.
//

#ifndef CDBAYES_DISTRIBUTEDDBNPATHTRAINER_H
#define CDBAYES_DISTRIBUTEDDBNPATHTRAINER_H

#include "BNPathTrainer.h"

namespace RegularizedNetwork {
    namespace Distributed {
        class DBNPathTrainer : public BNPathTrainer {
        protected:
            void spawnWorker() override;
            void managePool(PathTrainResult &result, int eras, int size,
                                                                             const std::vector<double> &lambdas) override;

        public:
            DBNPathTrainer(const Dataset &dataset, const Eigen::MatrixXd &weights,
                           const std::string &stateFilename = "") : BNPathTrainer(dataset, weights, stateFilename) {};
        };
    }
}

#endif //CDBAYES_DISTRIBUTEDDBNPATHTRAINER_H

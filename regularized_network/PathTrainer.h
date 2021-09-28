//
// Created by João Santos on 21/11/2020.
//

#ifndef CDBAYES_PATHTRAINER_H
#define CDBAYES_PATHTRAINER_H

#include "PathTrainResult.h"

#include <memory>

#define DEFAULT_MAX_CD_ITERS 500

namespace RegularizedNetwork {
    class PathTrainer {
    public:
        virtual std::unique_ptr<PathTrainResult> train(const std::vector<double> &lambdas, int eras) = 0;
        virtual std::unique_ptr<PathTrainResult> train(int maxEdges, double initialLambda, double step, int maxLambdas) = 0;
    };
}

#endif //CDBAYES_PATHTRAINER_H

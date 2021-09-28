//
// Created by João Santos on 17/03/2021.
//

#ifndef SPARSEDBN_HOTSTART_H
#define SPARSEDBN_HOTSTART_H

#include <utility>

#include "regularized_network/FixedPathTrainResult.h"
#include "common/HillClimbingHotStart.h"

namespace Common {
    class HotStart {
    public:
        RegularizedNetwork::FixedPathTrainResult regularizedNetworkHotStart;
        HillClimbingHotStart hillClimbingHotStart;

        HotStart(RegularizedNetwork::FixedPathTrainResult rnhot, HillClimbingHotStart hchot) :
            regularizedNetworkHotStart(std::move(rnhot)), hillClimbingHotStart(std::move(hchot)) {};
    };
}

#endif //SPARSEDBN_HOTSTART_H

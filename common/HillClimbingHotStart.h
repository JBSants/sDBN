//
// Created by João Santos on 17/03/2021.
//

#ifndef SPARSEDBN_HILLCLIMBINGHOTSTART_H
#define SPARSEDBN_HILLCLIMBINGHOTSTART_H

#include <vector>
#include "common/digraph.h"

namespace Common {
    struct HillClimbingHotStart {
        std::vector<double> lambdas;
        std::vector<Graph::Digraph> directed;
    };
}

#endif //SPARSEDBN_HILLCLIMBINGHOTSTART_H

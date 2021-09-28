//
// Created by João Santos on 15/03/2021.
//

#ifndef SPARSEDBN_IO_H
#define SPARSEDBN_IO_H

#include "regularized_network/PathTrainResult.h"
#include "regularized_network/FixedPathTrainResult.h"
#include "common/digraph.h"
#include "common/HillClimbingHotStart.h"
#include "common/HotStart.h"
#include <iostream>

namespace Common {
    namespace IO {
        void SerializePathTrainResult(std::ostream &output, const RegularizedNetwork::PathTrainResult &result);
        RegularizedNetwork::FixedPathTrainResult UnserializePathTrainResult(std::istream &input);
        void SerializeDigraph(std::ostream &output, const Graph::Digraph &digraph);
        Graph::Digraph UnserializeDigraph(std::istream &input);
        void SerializeHillClimbingHotStart(std::ostream &output, const HillClimbingHotStart &hotstart);
        HillClimbingHotStart UnserializeHillClimbingHotStart(std::istream &input);
        void SerializeHotStart(std::ostream &output, const HotStart &hotstart);
        HotStart UnserializeHotStart(std::istream &input);
    }
}

#endif //SPARSEDBN_IO_H

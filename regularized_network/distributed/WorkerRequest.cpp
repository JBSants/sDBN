//
// Created by João Santos on 27/12/2020.
//

#include "WorkerRequest.h"

namespace RegularizedNetwork {
    namespace Distributed {
        std::ostream &operator<<(std::ostream &out, WorkerRequest const &t) {
            switch (t.type) {
                case WorkerRequest::NEW_LAMBDA:
                    out << "WorkerRequest(type=NEW_LAMBDA,";
                    break;
                case WorkerRequest::TERMINATE:
                    out << "WorkerRequest(type=TERMINATE,";
                    break;
            }

            out << "rv=" << t.rv << ",lambda=" << t.lambda << ")";

            return out;
        }
    }
}
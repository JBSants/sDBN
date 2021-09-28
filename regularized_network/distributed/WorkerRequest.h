//
// Created by João Santos on 26/12/2020.
//

#ifndef CDBAYES_REGULARIZEDGRAPHTRAINERWORKERREQUEST_H
#define CDBAYES_REGULARIZEDGRAPHTRAINERWORKERREQUEST_H

#include <iostream>

namespace RegularizedNetwork {
    namespace Distributed {
        class WorkerRequest {
        public:
            enum RequestType {
                NEW_LAMBDA, TERMINATE
            };

            RequestType type;
            int rv;
            double lambda;

            WorkerRequest(RequestType type, int rv, double lambda) : type(type), rv(rv), lambda(lambda) {}

            WorkerRequest(RequestType type = TERMINATE)  : type(type), rv(-1), lambda(0) {}

            friend std::ostream & operator<< (std::ostream &out, WorkerRequest const &t);
        };


    }
}


#endif //CDBAYES_REGULARIZEDGRAPHTRAINERWORKERREQUEST_H

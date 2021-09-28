//
// Created by João Santos on 15/11/2020.
//

#ifndef CDBAYES_MULTINOMIALLOGISTICREGRESSION_H
#define CDBAYES_MULTINOMIALLOGISTICREGRESSION_H

#include <Eigen/Eigen>
#include <memory>
#include <ctime>

struct MultinomialLogisticRegression {
private:
    bool verbose = false;
public:
    double delta = 0;
    double epsilon = 1e-5;

    MultinomialLogisticRegression() = default;
    explicit MultinomialLogisticRegression(bool verbose) : verbose(verbose) {};
    void train(const Eigen::MatrixXd &data, const Eigen::MatrixXd &y, Eigen::MatrixXd &coeff) const;
};


#endif //CDBAYES_MULTINOMIALLOGISTICREGRESSION_H

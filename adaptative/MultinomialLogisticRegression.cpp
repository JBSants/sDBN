//
// Created by João Santos on 15/11/2020.
//

#include "MultinomialLogisticRegression.h"
#include "lbfgs.h"

#include <iostream>

using Eigen::MatrixXd;

struct lbgfs_problem {
    const Eigen::MatrixXd *data;
    const Eigen::MatrixXd *y;
    MatrixXd *p = nullptr;
    int gradCount = 0;
    double gradMed = 0;
};

static lbfgsfloatval_t evaluate(
        void *instance,
        const lbfgsfloatval_t *x,
        lbfgsfloatval_t *g,
        const int n,
        const lbfgsfloatval_t step
)
{
    lbfgsfloatval_t fx = 0.0;
    struct lbgfs_problem &problem = *((struct lbgfs_problem *) instance);
    const MatrixXd &data = *problem.data;
    const MatrixXd &y = *problem.y;
    const int rows = data.rows();
    const int cols = data.cols();
    const int ri = y.cols();
    Eigen::Map<const MatrixXd> coeff(x, cols, ri);
    Eigen::Map<MatrixXd> grad(g, cols, ri);
    MatrixXd &p = *problem.p;

    p.noalias() = data * coeff;

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < ri; j++) {
            if (y(i, j) != 0.0) {
                fx += p(i, j);
                break;
            }
        }
    }

    p = p.array().exp();

    for (int i = 0; i < rows; i++) {
        double rowSum = p.row(i).sum();
        fx -= log(rowSum);
        p.row(i) = y.row(i) - p.row(i) / rowSum;
    }

    for (int i = 0; i < ri; i++) {
        grad.col(i) = -(p.col(i).transpose() * data);
    }

    grad(cols-1, 0) = 0;
    grad /= rows;

    return -fx / rows;
}

static lbfgsfloatval_t evaluate_dumb(
        void *instance,
        const lbfgsfloatval_t *x,
        lbfgsfloatval_t *g,
        const int n,
        const lbfgsfloatval_t step
)
{
    int i;
    lbfgsfloatval_t fx = 0.0;

    for (i = 0;i < n;i += 2) {
        lbfgsfloatval_t t1 = 1.0 - x[i];
        lbfgsfloatval_t t2 = 10.0 * (x[i+1] - x[i] * x[i]);
        g[i+1] = 20.0 * t2;
        g[i] = -2.0 * (x[i] * g[i+1] + t1);
        fx += t1 * t1 + t2 * t2;
    }

    return fx;
}

static int progress(
        void *instance,
        const lbfgsfloatval_t *x,
        const lbfgsfloatval_t *g,
        const lbfgsfloatval_t fx,
        const lbfgsfloatval_t xnorm,
        const lbfgsfloatval_t gnorm,
        const lbfgsfloatval_t step,
        int n,
        int k,
        int ls
)
{
    printf("Iteration %d:\n", k);
    printf("  fx = %f, x[0] = %f, x[1] = %f\n", fx, x[0], x[1]);
    printf("  xnorm = %f, gnorm = %f, step = %f\n", xnorm, gnorm, step);
    printf("\n");
    return 0;
}

void MultinomialLogisticRegression::train(const Eigen::MatrixXd &data, const Eigen::MatrixXd &y, Eigen::MatrixXd &coeff) const {
    const int cols = data.cols();
    const int states = y.cols();
    lbfgsfloatval_t *x = lbfgs_malloc(cols*states);
    lbfgsfloatval_t fx;
    lbfgs_parameter_t param;
    MatrixXd probs(cols, states);

    for (int i = 0; i < cols*states; i++) {
        x[i] = 0.0;
    }

    struct lbgfs_problem p = { &data, &y };
    p.p = &probs;

    lbfgs_parameter_init(&param);
    param.delta = delta;
    param.epsilon = epsilon;

    int ret = lbfgs(cols*states, x, &fx, evaluate, verbose ? progress : nullptr, &p, &param);

    if (verbose) {
        printf("L-BFGS optimization terminated with status code = %d\n", ret);
        printf("  fx = %f, x[13] = %f, x[20] = %f\n", fx, x[13], x[20]);
    }

    std::cerr << "Weight regression status: " << ret << std::endl;

    coeff = Eigen::Map<Eigen::MatrixXd>(x, cols, states);

    lbfgs_free(x);
}

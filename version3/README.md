# SOBO
Single-Objective Bayesian Optimization

This version uses the BFGS algorithm to optimize the likelihood function, and a derivative-free global optimization algorithm CMAES(Covariance Matrix Adaptation Evolution Strategy) is utilized to optimize the acquisition function.

Since the complexity of GP model training is $O(N^3)$, time cost for training will become unacceptable if the data size is large. A parallel version of BFGS algorithm is used here to reduce the time for practical considerations.


Dependencies:

Cma: https://github.com/CMA-ES/pycma

Optimparallel: https://github.com/florafauna/optimParallel-python

Scipy: https://github.com/scipy/scipy


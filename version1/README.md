# SOBO
Single-Objective Bayesian Optimization

This version uses the BFGS algorithm to optimize the likelihood function, and multiple starting point (MSP) strategy is utilized to optimize the acquisition function.
Since BFGS depends on the gradient value, in order to improve the speed and accuracy of calculating the gradient, an package named Autograd is used here.

It looks better than other methods, but the flexibility of Autograd is not very good.

Dependencies:

Autograd: https://github.com/HIPS/autograd

Scipy: https://github.com/scipy/scipy


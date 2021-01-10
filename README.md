# SOBO

## About
Single-Objective Bayesian Optimization

An efficient global optimization algorithm for constrained (or not) single-objective problems.
It is usually used to optimize the expensive black box functions because of its high sampling efficiency, but not suitable for situations with high-dimensional search space.

The optimization process for one toy function is shown here.

![image](https://github.com/Xiao-dong-Wang/SOBO/blob/master/version2/figures/BO_it_1.png)

![image](https://github.com/Xiao-dong-Wang/SOBO/blob/master/version2/figures/BO_it_5.png)

![image](https://github.com/Xiao-dong-Wang/SOBO/blob/master/version2/figures/BO_it_9.png)

Here are three versions of SOBO. Version 1 describes the GP training and Bayesian optimization process in detail. Version 2 directly uses the GPy module for GP modeling. Both of them use the gradient algorithm to optimize the acquisition function. Version 3 uses Evolutionary algorithm to optimize the acquisition function.


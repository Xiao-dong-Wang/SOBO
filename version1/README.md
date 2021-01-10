# SOBO

## About
Single-Objective Bayesian Optimization

This version uses the BFGS algorithm for GP model training. A multiple-starting-point (MSP) strategy is utilized to optimize the acquisition function, weighted Expected Improvement (wEI).
Autograd is used to improve the speed and accuracy of gradient calculation.

The optimization process for one toy function is shown here.

![image](https://github.com/Xiao-dong-Wang/SOBO/blob/master/version1/figures/BO_it_1.png)

![image](https://github.com/Xiao-dong-Wang/SOBO/blob/master/version1/figures/BO_it_4.png)

![image](https://github.com/Xiao-dong-Wang/SOBO/blob/master/version1/figures/BO_it_7.png)

The sampling efficiency of this method is very high, but the flexibility of Autograd is not that good.

## Usage
See **run.py**

```
import autograd.numpy as np
import sys
import toml
from src.util import *
from src.BO import BO
from get_dataset import *
import multiprocessing
import pickle

argv = sys.argv[1:]
conf = toml.load(argv[0])

name = conf['funct']
funct = get_funct(name)
num = conf['num']
bounds = np.array(conf['bounds'])
bfgs_iter = conf['bfgs_iter']
iteration = conf['iteration']
K = conf['K']

data = init_dataset(funct, num, bounds)
dataset = {}
dataset['train_x'] = data['train_x']
dataset['train_y'] = data['train_y']

with open('dataset.pickle', 'wb') as f:
    pickle.dump(dataset, f)


for ii in range(iteration):
    print('********************************************************************')
    print('iteration',ii)
    model = BO(dataset, bounds, bfgs_iter, debug=False)
    best_x = model.best_x
    best_y = model.best_y
    print('best_x', best_x)
    print('best_y', best_y)

    p = np.minimum(int(K/5), 5)
    def task(x0):
        x0 = model.optimize_constr(x0)
        x0 = model.optimize_wEI(x0)
        wEI_tmp = model.calc_log_wEI_approx(x0)
        return x0, wEI_tmp

#   Acquisition function optimization via MSP method
    pool = multiprocessing.Pool(processes=5)
    x0_list = []
    for i in range(int(K/p)):
        x0_list.append(model.rand_x(p))
    results = pool.map(task, x0_list)
    pool.close()
    pool.join()

    candidate = results[0][0]
    wEI_tmp = results[0][1]
    for j in range(1, int(K/p)):
        candidate = np.concatenate((candidate.T, results[j][0].T)).T
        wEI_tmp = np.concatenate((wEI_tmp.T, results[j][1].T)).T

    idx = np.argsort(wEI_tmp)[-1:]
    new_x = candidate[:, idx]
    new_y = funct(new_x, bounds)

    print('idx',idx)
    print('x',new_x.T)
    print('y',new_y.T)
    dataset['train_x'] = np.concatenate((dataset['train_x'].T, new_x.T)).T
    dataset['train_y'] = np.concatenate((dataset['train_y'].T, new_y.T)).T
    with open('dataset.pickle', 'wb') as f:
        pickle.dump(dataset, f)
```


## Dependencies:

Autograd: https://github.com/HIPS/autograd

Scipy: https://github.com/scipy/scipy


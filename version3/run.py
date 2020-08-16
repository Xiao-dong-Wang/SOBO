import autograd.numpy as np
import sys
import toml
from src.util import *
from src.BO import BO
from get_dataset import *
import multiprocessing
import pickle
import time

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

    x0 = model.optimize_constr()
    new_x = model.optimize_wEI(x0.reshape(-1))
    new_y = funct(new_x, bounds)

    print('x',new_x.T)
    print('y',new_y.T)
    dataset['train_x'] = np.concatenate((dataset['train_x'].T, new_x.T)).T
    dataset['train_y'] = np.concatenate((dataset['train_y'].T, new_y.T)).T
    with open('dataset.pickle', 'wb') as f:
        pickle.dump(dataset, f)




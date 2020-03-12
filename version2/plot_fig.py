import numpy as np
import sys
import toml
from src.util import *
from src.BO import BO
from get_dataset import *
import multiprocessing
import pickle
import matplotlib.pyplot as plt

np.random.seed(1234)

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

# Test data
nn = 200
X_star = np.linspace(-0.5, 0.5, nn)[None,:]
y_star = funct(X_star,bounds)
X_star_real = X_star * (bounds[0,1]-bounds[0,0]) + (bounds[0,1]+bounds[0,0])/2

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

    y_pred, y_var = model.predict(X_star)
    train_x = dataset['train_x']
    train_y = dataset['train_y']
    train_x_real = train_x * (bounds[0,1]-bounds[0,0]) + (bounds[0,1]+bounds[0,0])/2
    
    p = np.minimum(int(K/5), 5)
    def task(x0):
        x0 = model.optimize_wEI(x0)
        wEI_tmp = model.calc_wEI(x0)
        return x0, wEI_tmp

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


    EI = model.calc_wEI(X_star)
    new_x_real = new_x * (bounds[0,1]-bounds[0,0]) + (bounds[0,1]+bounds[0,0])/2

    plt.figure(1)
    plt.subplot(2,1,1)
    plt.cla()
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif', size=10)
    plt.plot(X_star_real.flatten(), y_star.flatten(), 'b-', label = "Exact", linewidth=2)
    plt.plot(X_star_real.flatten(), y_pred.flatten(), 'r--', label = "Prediction", linewidth=2)
    lower = y_pred - 2.0*np.sqrt(y_var)
    upper = y_pred + 2.0*np.sqrt(y_var)
    plt.fill_between(X_star_real.flatten(), lower.flatten(), upper.flatten(), 
                     facecolor='orange', alpha=0.5, label="Two std band")
    plt.plot(train_x_real, train_y, 'bo', label = "Data")
    ax = plt.gca()
    ax.set_xlim([bounds[0,0], bounds[0,1]])
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.title("Iteration %d" % (ii+1))

    # Plot wEI
    plt.subplot(2,1,2)
    plt.cla()
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif', size=10)
    plt.plot(X_star_real.flatten(), EI, 'b-', linewidth=2)
    ax = plt.gca()
    ymin, ymax = ax.get_ylim()
    plt.plot(new_x_real.flatten()*np.ones(2), np.linspace(ymin,ymax,2),'k--')
    ax.set_xlim([bounds[0,0], bounds[0,1]])
    plt.xlabel('x')
    plt.ylabel('EI(x)')
    plt.pause(1.0)
    plt.savefig("./figures/BO_it_%d.png" % (ii+1), format='png',dpi=100)









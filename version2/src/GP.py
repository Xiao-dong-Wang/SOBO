import numpy as np
import GPy

# Constructing GP model based on GPy module 
class GP:
    # Initialize GP class
    def __init__(self, dataset):
        self.train_x = dataset['train_x'].T
        self.train_y = dataset['train_y'].T
        self.dim = self.train_x.shape[1]
        self.num_train = self.train_x.shape[0]
        self.normalize()

    # Normalize y
    def normalize(self):
        self.train_y = self.train_y.reshape(-1)
        self.mean = self.train_y.mean()
        self.std = self.train_y.std() + 0.000001
        tmp = (self.train_y - self.mean)/self.std
        self.train_y = tmp[:,None]


    # GP model training
    # Needed by GPy: train_x shape: (num_train, dim);   train_y shape: (num_train, 1) 
    def train(self):
        k = GPy.kern.RBF(self.dim, ARD=True)
        m = GPy.models.GPRegression(X=self.train_x, Y=self.train_y, kernel=k)
        m.kern.variance = np.var(self.train_y)
        m.kern.lengthscale = np.std(self.train_x)
        m.likelihood.variance = 0.01 * np.var(self.train_y)
        m.optimize()
        self.model = m
        print('GP. GP model training process finished')

    def predict(self, test_x):
        test_x = test_x.T
        py, ps2 = self.model.predict(test_x)
        py = py * self.std + self.mean
        ps2 = ps2 * (self.std**2)
        py = py.reshape(-1)
        ps2 = ps2.reshape(-1)
        return py, ps2
    

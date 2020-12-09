from tensorflow.keras import utils
from tensorflow.keras.preprocessing import image
import numpy as np

def add_extra_feature(x):
    x_with_extra_feature = np.zeros((x.shape[0],x.shape[1]+1))
    x_with_extra_feature[:,:-1] = x
    x_with_extra_feature[:,-1] = int(1)
    return x_with_extra_feature

class SVM_Classifier(object):
    def __init__(self, etha=0.01, alpha=0.1, epochs=200):
        self.epochs = epochs
        self.etha = etha
        self.alpha = alpha
        self.theta = None
        self.train_cost_func = None

    def learning(self, X_train, Y_train): #arrays: Y =-1,1

        if len(set(Y_train)) != 2:
            raise ValueError("Number of classes in Y has to be = 2!")

        X_train = add_extra_feature(X_train)
        self.theta = np.random.normal(loc=0, scale=0.05, size=X_train.shape[1])
        train_cost_func_epoch = []

        for epoch in range(self.epochs):
            train_cost_func = 0
            for i,x in enumerate(X_train):
                margin = Y_train[i]*np.dot(self.theta,X_train[i])
                if margin >= 1: # классифицируем верно
                    self.theta = self.theta - self.etha*self.alpha*self.theta/self.epochs
                    train_cost_func += self.soft_margin_func(X_train[i],Y_train[i])
                else: # классифицируем неверно или попадаем внутрь разделяющей полосы
                    self.theta = self.theta +\
                    self.etha*(Y_train[i]*X_train[i] - self.alpha*self.theta/self.epochs)
                    train_cost_func += self.soft_margin_func(X_train[i],Y_train[i])
            train_cost_func_epoch.append(train_cost_func)
        self.train_cost_func = np.array(train_cost_func_epoch)
        return self.theta

    def prediction(self, X:np.array, th) -> np.array:
        y_pred = []
        X_extended = add_extra_feature(X)
        for i in range(len(X_extended)):
            y_pred.append(np.sign(np.dot(th,X_extended[i])))
        return np.array(y_pred)

    def hinge_func(self, x, y):
        return max(0,1 - y*np.dot(x, self.theta))

    def soft_margin_func(self, x, y):
        return self.hinge_func(x,y)+self.alpha*np.dot(self.theta, self.theta)

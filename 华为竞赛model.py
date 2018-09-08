# coding: utf-8
from functools import reduce
import random
import math

import metric

class Regressor(object):
    """ Regression method for training a model to predict flavot demands
    """
    def __init__(self, alpha=0.001, learningRate=0.01, randomSeed=2018, numEpoches=10000, epsilon=1.e-5, sampleWeights=None):
        """
        Initialize some parameters for the regressor

        Parameters
        ----------
        alpha : {int, float}
                L2
        learningRate : {int, float}
                       learning rates
        randomSeed: {int}
                    random seed
        
        Returns
        -------
        class Regressor
        """
        self.alpha = alpha
        self.learningRate = learningRate
        self.predictorNum = None
        self.observeNum = None
        self.weights = None
        self.randomSeed = randomSeed
        self.numEpoches = numEpoches
        self.epsilon = epsilon
        self.sampleWeights = sampleWeights
        
    def fit(self, X, y):
        """ Fit data
        """
        N = len(y)
        if not (len(X) == N):
            raise RuntimeError("X's length %d dosen't match y's length %d" % (len(X), N))
        self.predictorNum = len(X[0])
        self.observeNum = 1 if isinstance(y[0], (float, int)) else len(y[0])
        random.seed(a=self.randomSeed)
        self.weights = [random.random() for _ in xrange(self.predictorNum + 1)]

        y_mrs = mrs(y)
        yHat = self.predict(X)
        it = 0
        mse_ = 0
        for epoch in xrange(self.numEpoches):
            yHat_ = yHat[:]
            yHat_mrs = mrs(yHat)
            error = map(lambda x: x[0] - x[1], zip(y, yHat))
            error_mrs = mrs(error)
            for sample in xrange(N):
                DlDy = scheer(y_mrs, yHat_mrs, error_mrs, y, yHat, error, sample)
                for s in xrange(self.predictorNum + 1):
                    if s == self.predictorNum:
                        self.weights[s] -= self.learningRate*DlDy/N
                    else:
                        if self.sampleWeights is None:
                            self.weights[s] -= self.learningRate*DlDy/N
                        else:
                            self.weights[s] -= self.learningRate*DlDy/N*self.sampleWeights[sample]               
            
            yHat = self.predict(X)
            if epoch%500 == 0:
                print u"Epoch {0}: mse - {1}".format(epoch, metric._mse(yHat, y))
            if epoch%20 == 0:
                mse = metric._mse(yHat, y)
                if mse - mse_ >= 0.0:
                    self.learningRate *= 0.8
                mse_ = mse

            yHatDeltaNorm = sum(map(lambda x: (x[0] - x[1])**2., zip(yHat_, yHat)))
            yHatDeltaNorm = math.sqrt(yHatDeltaNorm)

            if (yHatDeltaNorm < self.epsilon) or (epoch == self.numEpoches - 1):
                it += 1
                if (it > 10) or (epoch == self.numEpoches - 1):
                    mse = metric._mse(y, yHat)
                    print u"Epoch [{0}]: mse - {1}".format(epoch, mse)
                    break
    
    def predict(self, X):
        """ Predictor for X
        """
        if (len(X) == self.predictorNum) and isinstance(X[0], (int, float)):
            X = [X]
        elif not (len(X[0]) == self.predictorNum):
            raise RuntimeError('Input X must have %d column(s), but it has %d column(s)' % (self.predictorNum, len(X[0])))
        
        preds = []
        for sample in xrange(len(X)):
            demands = 0.
            for s in xrange(self.predictorNum + 1):
                x = 1. if s == self.predictorNum else X[sample][s]
                demands += self.weights[s] * x
            preds.append(demands)
        
        return preds

class Regressor2(object):
    """ Regression method for training a model to predict flavot demands
    """
    def __init__(self, alpha=0.001, learningRate=0.01, randomSeed=2018, numEpoches=10000, epsilon=1.e-5, sampleWeights=None):
        """
        Initialize some parameters for the regressor

        Parameters
        ----------
        alpha : {int, float}
                L2
        learningRate : {int, float}
                       learning rates
        randomSeed: {int}
                    random seed
        
        Returns
        -------
        class Regressor
        """
        self.alpha = alpha
        self.learningRate = learningRate
        self.predictorNum = None
        self.observeNum = None
        self.weights = None
        self.randomSeed = randomSeed
        self.numEpoches = numEpoches
        self.epsilon = epsilon
        self.sampleWeights = sampleWeights
        
    def fit(self, X, y):
        """ Fit data
        """
        N = len(y)
        if not (len(X) == N):
            raise RuntimeError("X's length %d dosen't match y's length %d" % (len(X), N))
        self.predictorNum = len(X[0])
        self.observeNum = 1 if isinstance(y[0], (float, int)) else len(y[0])
        random.seed(a=self.randomSeed)
        self.weights = [random.random() for _ in xrange(self.predictorNum + 1)]

        # weights_ = self.weights[:]
        y_mrs = mrs(y)
        yHat = self.predict(X)
        it = 0
        for epoch in xrange(self.numEpoches):
            yHat_ = yHat[:]
            yHat_mrs = mrs(yHat)
            error = map(lambda x: x[0] - x[1], zip(y, yHat))
            error_mrs = mrs(error)
            for sample in xrange(N):
                if self.sampleWeights is None:
                    DlDy = scheer(y_mrs, yHat_mrs, error_mrs, y, yHat, error, sample)
                else:
                    DlDy = scheer(y_mrs, yHat_mrs, error_mrs, y, yHat, error, sample)*self.sampleWeights[sample]
                for s in xrange(self.predictorNum + 1):
                    x = 1. if s == self.predictorNum else X[sample][s]
                    self.weights[s] -= self.learningRate*DlDy*x/N
                    if self.alpha > 0.0:
                        self.weights[s] -= self.alpha*self.weights[s]/N
            
            yHat = self.predict(X)
            if epoch%100 == 0:
                print u"Epoch {0}: mse - {1}".format(epoch, metric._mse(yHat, y))

            # weightsDeltaNorm = reduce(lambda x, y: x + y, map(lambda x: (x[0] - x[1])**2., zip(weights_, self.weights)))
            # weightsDeltaNorm = math.sqrt(weightsDeltaNorm)
            # if (weightsDeltaNorm < 0.001):
            #     self.learningRate *= 0.9
            yHatDeltaNorm = sum(map(lambda x: (x[0] - x[1])**2., zip(yHat_, yHat)))
            yHatDeltaNorm = math.sqrt(yHatDeltaNorm)
            if (yHatDeltaNorm < 0.001):
                self.learningRate *= 0.99

            # if (weightsDeltaNorm < self.epsilon) or (epoch == self.numEpoches - 1):
            #     it += 1
            #     if (it > 10) or (epoch == self.numEpoches - 1):
            #         mse = metric._mse(y, yHat)
            #         print u"Epoch [{0}]: mse - {1}".format(epoch, mse)
            #         break
            # weights_ = self.weights[:]
            if (yHatDeltaNorm < self.epsilon) or (epoch == self.numEpoches - 1):
                it += 1
                if (it > 10) or (epoch == self.numEpoches - 1):
                    mse = metric._mse(y, yHat)
                    print u"Epoch [{0}]: mse - {1}".format(epoch, mse)
                    break
    
    def predict(self, X):
        """ Predictor for X
        """
        if (len(X) == self.predictorNum) and isinstance(X[0], (int, float)):
            X = [X]
        elif not (len(X[0]) == self.predictorNum):
            raise RuntimeError('Input X must have %d column(s), but it has %d column(s)' % (self.predictorNum, len(X[0])))
        
        preds = []
        for sample in xrange(len(X)):
            demands = 0.
            for s in xrange(self.predictorNum + 1):
                x = 1. if s == self.predictorNum else X[sample][s]
                demands += self.weights[s] * x
            preds.append(demands)
        
        return preds

def double_smoothing(y, alpha):
    S2_1 = []
    S2_2 = []

    x = 0
    for n in xrange(3):
        x += float(y[n])
    x = x / 3

    S2_1.append(x)
    S2_2.append(x)

    ##下面是计算一次指数平滑的值
    S2_1_new = []
    for j in xrange(len(y)):
        if j == 0:
            S2_1_new.append(float(alpha) * float(y[j]) + (1 - float(alpha)) * float(S2_1[j]))
        else:
            S2_1_new.append(float(alpha) * float(y[j]) + (1 - float(alpha)) * float(S2_1_new[-1]))

    ##下面是计算二次指数平滑的值    
    S2_2_new = []
    for j in xrange(len(y)):
        if j == 0:
            S2_2_new.append(float(alpha) * float(S2_1_new[j]) + (1 - float(alpha)) * float(S2_2[j]))
        else:
            S2_2_new.append(float(alpha) * float(S2_1_new[j]) + (1 - float(alpha)) * float(S2_2_new[-1]))  ##计算二次指数的值
    
    print u"Double smoothing MSE: {0}".format(metric._mse(y, S2_2_new))
    # print y
    # print S2_2_new

    ##下面是计算At、Bt以及每个预估值Xt的值，直接计算预估值，不一一列举Xt的值了
    At = (float(S2_1_new[-1]) * 2 - float(S2_2_new[-1]))
    Bt = (float(alpha) / (1 - float(alpha)) * (float(S2_1_new[-1]) - float(S2_2_new[-1])))
    return At + Bt

def mrs(x):
    N = len(x)
    return math.sqrt(sum(map(lambda x: x**2., x))/N)

def scheer(y_mrs, yHat_mrs, error_mrs, y, yHat, error, sample):
    N = len(y)
    part1 = -error_mrs*2*yHat[sample] / (2*N*yHat_mrs*(y_mrs + yHat_mrs)**2.)
    part2 = -2*(error[sample]) / (2*N*error_mrs*(y_mrs + yHat_mrs))
    return part1 + part2
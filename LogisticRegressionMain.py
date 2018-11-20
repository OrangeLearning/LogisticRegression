"""
    This file is realize the logistic regression algorithm

    the x vector should be the format of [x1 ,x2 ,x3 ,...,xn].T
"""
from LogisticRegressionData import getWatermelonData
import numpy as np
import math

EPS = 0.001

def get_w_bByBeta(beta):
    w = beta[:-1]
    b = beta[-1]
    return w , np.array([b])

def getBetaBy_w_b(w, b):
    print(w.shape)
    print(b.shape)
    return np.concatenate((w,b) ,axis=0)

def prepareData():
    data = getWatermelonData()
    y = []
    xs = []
    for item in data:
        y.append([item[-1]])
        xs.append(item[:-1])
    y = np.array(y)
    xs = np.array(xs)

    return xs , y

def fun(x, w, b):
    """
        :param x: (n , 1)
        :param w: (n , 1)
        :param b: (1 , 1)
        :return:
    """
    return np.dot(w.T, x) + b

def p1(x, w, b):
    return 1.0 / (1.0 + np.exp(fun(x, w, b)))


def p0(x, w, b):
    return 1 - p1(x, w, b)


def derivativeOne(beta ,xs , ys):
    w ,b = get_w_bByBeta(beta)
    newBeta = np.zeros(beta.shape)

    for i in range(len(xs)):
        print("xs[i].shape = ", xs[i].shape)
        v = ys[i] - p1(xs[i][:-1] , w, b)
        print("v = ",v)
        newBeta -= v * xs[i]

    print("newBeta = ",newBeta)
    return newBeta

def derivativeTwo(beta , xs, ys):
    w ,b = get_w_bByBeta(beta)
    res = 0.0
    for item in xs:
        v1 = p1(item[:-1] ,w, b) * p0(item[:-1] , w, b)
        v2 = np.dot(item[:-1].T, item[:-1])
        res += (v1 * v2)
    return res

def loss(x1, x2):
    if x1.shape != x2.shape:
        raise ValueError("x1.shape and x2.shape is not same")
    delta_x = x1 - x2
    res = 0.0
    for item in delta_x:
        res += (item)**2
    return math.sqrt(res)

def NewtonMethod(x_s , ys, eps):
    n,m = x_s.shape
    beta = np.ones((m,1))
    print("beta = ",beta)
    while True:
        newBeta = beta - (1 / derivativeTwo(beta , x_s ,ys)) * derivativeOne(beta , x_s,ys)

        if loss(beta , newBeta) < eps:
            beta = np.copy(newBeta)
            break
        print("loss = " ,loss(beta ,newBeta))
        beta = np.copy(newBeta)
    return beta

def LogisticRegression(xs ,ys):
    n , m = xs.shape
    x_s = np.zeros((n , m+1))
    for i in range(len(xs)):
        for j in range(len(xs[i])):
            x_s[i][j] = xs[i][j]
        x_s[i][-1] = 1.0

    beta = NewtonMethod(x_s , ys, EPS)
    w ,b = get_w_bByBeta(beta)
    print(w, b)

    return w,b

def main():
    xs ,y = prepareData()
    LogisticRegression(xs ,y)

def test():
    xs = [
        np.array([[1],[2],[1]]),
        np.array([[10],[20],[1]])
    ]

    ys = [
        np.array([[1]]),
        np.array([[2]])
    ]

    beta = np.array([[3],[2],[1]])
    derivativeOne(beta , xs,ys)

    print("test finish")

if __name__ == '__main__':
    main()

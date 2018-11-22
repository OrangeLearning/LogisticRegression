"""
    This file is realize the logistic regression algorithm

    the x vector should be the format of [x1 ,x2 ,x3 ,...,xn].T
"""
import matplotlib.pyplot as plt
from LogisticRegressionData import getWatermelonData
import numpy as np
import math
from sys import maxsize

EPS = 0.001


def get_w_bByBeta(beta):
    w = beta[:-1]
    b = beta[-1]
    return w, b


def getBetaBy_w_b(w, b):
    # print("w.shape = ", w.shape)
    # print("b.shape = ", b.shape)
    return np.concatenate((w, b), axis=0)


def prepareData():
    data = getWatermelonData()
    y = []
    xs = []
    for item in data:
        y.append(item[-1])
        xs.append(item[:-1])
    y = np.array(y)
    xs = np.array(xs)
    # print("prepare xs.shape = ",xs.shape)
    # print("prepare y.shape = ",y.shape)
    return xs, y


def fun(x, w, b):
    """
        :param x: (n , 1)
        :param w: (n , 1)
        :param b: (1 , 1)
        :return:
    """
    return np.dot(w.T, x) + b


def p0(x, w, b):
    # beta = getBetaBy_w_b(w,b)
    # print("in p1 x.shape = ",x.shape," w.shape = ",w.shape )
    return 1.0 / (1.0 + math.exp(fun(x, w, b)))


def p1(x, w, b):
    return 1 - p0(x, w, b)


def derivativeOne(beta, xs, ys):
    w, b = get_w_bByBeta(beta)
    n = beta.shape
    newBeta = np.zeros(n)

    for i in range(len(xs)):
        # print("xs[i].shape = ", xs[i].shape)
        v = ys[i] - p1(xs[i][:-1], w, b)
        # print("v[0,0] = ", v)
        # print("xs[i].shape = ", xs[i].shape)
        # print("newBeta.shape = ", newBeta.shape)
        newBeta -= v * xs[i]

    # print("newBeta = ", newBeta)
    return newBeta


def derivativeTwo(beta, xs, ys):
    w, b = get_w_bByBeta(beta)
    res = 0.0
    for item in xs:
        # print("item = ", item)
        # print("fun = ",fun(item[:-1], w, b))
        v1 = p1(item[:-1], w, b) * p0(item[:-1], w, b)
        v2 = np.dot(item.T, item)
        res += (v1 * v2)
        # print("in dTwo : v1 = ", v1, " v2 = ", v2, " res = ", res)
    return res


def loss(x1, x2):
    if x1.shape != x2.shape:
        raise ValueError("x1.shape and x2.shape is not same")
    delta_x = x1 - x2
    res = 0.0
    for item in delta_x   :
        res += (item) ** 2
    # print("loss = ", delta_x, " res = ", math.sqrt(res))
    return math.sqrt(res)


def NewtonMethod(x_s, ys, eps):
    n, m = x_s.shape
    beta = np.ones((m,))
    print("beta = ", beta)
    while True:
        d2 = derivativeTwo(beta, x_s, ys)
        # print("d2 = ", d2)
        d1 = derivativeOne(beta, x_s, ys)
        # print("d1 = ", d1)

        newBeta = beta - (1 / d2) * d1
        print("in newton :\n", "\tnewBeta = ", newBeta, "\n\tbeta = ", beta)
        if loss(beta, newBeta) < eps:
            beta = np.copy(newBeta)
            break
        print("loss = ", loss(beta, newBeta))
        beta = np.copy(newBeta)
    return beta


def LogisticRegression(xs, ys):
    n, m = xs.shape
    x_s = np.zeros((n, m + 1))
    for i in range(len(xs)):
        for j in range(len(xs[i])):
            x_s[i][j] = xs[i][j]
        x_s[i][-1] = 1.0

    beta = NewtonMethod(x_s, ys, EPS)
    w, b = get_w_bByBeta(beta)
    print(w, b)

    return w, b


def showGraph(xs , ys , w, b):
    print("start show")

    x0_good_set = []
    x1_good_set = []
    x0_bad_set  = []
    x1_bad_set  = []

    x0_min = float(maxsize)
    x0_max = float(-maxsize)
    x1_min = float(maxsize)
    x1_max = float(-maxsize)
    for x,y in zip(xs,ys):
        if y == 1:
            x0_good_set.append(x[0])
            x1_good_set.append(x[1])
        else :
            x0_bad_set.append(x[0])
            x1_bad_set.append(x[1])

        x0_min = min(x0_min , x[0])
        x1_min = min(x1_min , x[1])
        x0_max = max(x0_max , x[0])
        x1_max = max(x1_max , x[1])

    plt.scatter(x0_good_set,x1_good_set,c = 'g',marker='*')
    plt.scatter(x0_bad_set,x1_bad_set,c = 'r',marker='o')
    # plt.show()

    good_set_x0 = []
    good_set_x1 = []
    bad_set_x0 = []
    bad_set_x1 = []

    x0 = x0_min

    while x0 < x0_max:
        x1 = x1_min
        while x1 < x1_max:
            y = x0 * w[0] + x1 * w[1] + b
            y = 1.0 / (1.0 + math.exp(-y))
            print("x0 = ",x0 , " x1 = ",x1 , " y = ",y , " int(y) = ",int(y + 0.5))
            if int(y + 0.5) == 1:
                good_set_x0.append(x0)
                good_set_x1.append(x1)
            else:
                bad_set_x0.append(x0)
                bad_set_x1.append(x1)

            x1 += 0.007
        x0 += 0.007

    print("len = ",len(good_set_x0))
    plt.scatter(good_set_x0,good_set_x1,marker= '.')
    plt.scatter(bad_set_x0,bad_set_x1,c = 'y',marker='.')
    plt.show()


    pass


def main():
    xs, y = prepareData()

    # print("xs = ",xs)
    # print("y = ",y)
    # for item in xs:
    #     print("item.shape = ",item.shape)
    w, b = LogisticRegression(xs, y)

    showGraph(xs, y , w , b)

def test():
    x = np.array([1, 2, 3])
    w = np.array([1, 2, 3])
    print("x.shape = ", x.shape)

    y = np.dot(w.T, x)
    print("y = ", y)
    print("test finish")


if __name__ == '__main__':
    # test()
    main()

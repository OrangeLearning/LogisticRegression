# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import math
# inputfile = 'xigua3.0.xlsx'
#
# # 导入数据
# data = pd.read_excel(inputfile, 'Sheet1')
# x = np.array([list(data[u'密度']), list(data[u'含糖率']), [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])
# x = x.T
# y = np.array([1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0])
#
# # 初始化参数
# beta = np.array([[0.01], [0.03], [1]])
# l_beta = 0
# old_l_beta = 0
# n = 0
# while True:
#     # 计算l(beta)
#
#     beta_T = np.transpose(beta)
#     # print(np.array([x[0,:]]).shape)
#     for i in np.arange(len(x)):
#         l_beta = l_beta + (-y[i] * np.dot(beta_T, np.array([x[i, :]]).T) + np.log(
#             1 + np.exp(np.dot(beta_T, np.array([x[i, :]]).T))))
#
#     if np.abs(l_beta - old_l_beta).any() <= 0.000001:
#         break
#     # 进行迭代
#     dbeta = 0
#     d2beta = 0
#     n = n + 1
#     old_l_beta = l_beta
#     for i in np.arange(len(x)):
#         x_i = np.array([x[i, :]])
#         x_i_2 = np.dot(x_i, x_i.T)
#         exp_b_x = np.exp(np.dot(np.transpose(beta), x_i.T))
#
#         dbeta = dbeta - np.array([x[i, :]]) * (y[i] - (exp_b_x / (1 + exp_b_x)))
#
#         d2beta = d2beta + x_i_2 * exp_b_x / ((1 + exp_b_x) * (1 + exp_b_x))
#     beta = beta - np.dot(np.linalg.inv(d2beta), dbeta).T
#
#     print("迭代次数=>", n)
#     print('模型参数=>', beta)


def main():

    pass

if __name__ == '__main__':
    main()
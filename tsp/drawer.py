#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

order_list = [
    [31, 52, 49, 47],
    [21, 17, 31, 51],
    [11, 36, 52, 27],
    [16, 8, 7, 27],
    [52, 58, 37, 38],
    [23, 25, 48, 56],
    [32, 47, 63, 62],
    [43, 69, 22, 35],
    [29, 32, 55, 28],
    [20, 47, 43, 62]
]
cityPositions = []
for order in order_list:
    cityPositions.append([order[2] + order[3], order[2], order[3]])
    cityPositions.append([order[2] + order[3] + 1000, order[2], order[3]])

plt.figure(1)  # 创建图表1
# x = np.linspace(0, 3, 100)
x = map(lambda cp: cp[1], cityPositions)
y = map(lambda cp: cp[2], cityPositions)
for i in xrange(5):
    plt.figure(1)  # ❶ # 选择图表1
    # plt.plot(x, np.exp(i * x / 3))
    plt.plot(x, y)
    # plt.sca(ax1)  # ❷ # 选择图表2的子图1
    # plt.plot(x, np.sin(i * x))
    # plt.sca(ax2)  # 选择图表2的子图2
    # plt.plot(x, np.cos(i * x))
plt.show()

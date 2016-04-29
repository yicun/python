#!/usr/bin/python
# -*- coding: utf-8 -*-
#
# 蚁群算法最早用来求解TSP问题
#

import sys
import getopt
import random
from math import sqrt
from math import pow


class Ant:
    tabu = []  # 禁忌表
    allowedCities = []  # 允许搜索的城市
    delta = [[]]  # 信息数变化矩阵
    distance = [[]]  # 距离矩阵
    alpha = 0.0
    beta = 0.0

    tourLength = float()  # 路径长度
    cityNum = 0  # 城市数量
    firstCity = 0  # 起始城市
    currentCity = 0  # 当前城市

    def __init__(self, cityNum=30, tourLength=0):
        self.cityNum = cityNum
        self.tourLength = tourLength

    def init(self, distance=[[]], alpha=0.0, beta=0.0):
        self.alpha = alpha
        self.beta = beta
        self.tabu = []
        self.distance = distance
        self.delta = [[0 for col in range(self.cityNum)] for row in range(self.cityNum)]
        self.allowedCities = range(0, self.cityNum, 1)

        self.firstCity = random.nextInt(self.cityNum)
        self.allowedCities.remove(self.firstCity)
        self.tabu.add(self.firstCity)
        self.currentCity = self.firstCity

    def selectNextCity(self, pheromone=[[]]):
        p = [0 for col in range(self.cityNum)]
        sumPh = 0.0
        # 计算分母部分
        for i in self.allowedCities:
            sumPh += pow(pheromone[self.currentCity][i], self.alpha) * pow(1.0 / self.distance[self.currentCity][i], self.beta)
        # 计算概率矩阵
        for i in range(0, self.cityNum, 0):
            flag = False
            for j in self.allowedCities:
                if i == j:
                    p[i] = pow(pheromone[self.currentCity][i], self.alpha) * pow(1.0 / self.distance[self.currentCity][i], self.beta) / sumPh
                    flag = True
                    break
            if not flag:
                p[i] = 0.0

        # 轮盘赌选择下一个城市
        selectP = random.random()
        selectCity = 0
        sum1 = 0.0
        for i in range(0, self.cityNum, 0):
            sum1 += p[i]
            if sum1 >= selectP:
                selectCity = i
                break

        # 从允许选择的城市中去除selectCity
        self.allowedCities.remove(selectCity)
        # 在禁忌表中添加selectCity
        self.tabu.add(selectCity)
        # 将当前城市改为选择的城市
        self.currentCity = selectCity

    def calculateTourLength(self):
        tourLen = 0
        for i in range(0, self.cityNum, 0):
            tourLen += self.distance[self.tabu[i]][self.tabu[i + 1]]
        return tourLen


class AntColonyOptimization(Exception):
    ants = []  # 蚂蚁
    antNum = 0  # 蚂蚁数量
    cityNum = 0  # 城市数量
    MAX_GEN = 0  # 运行代数
    pheromone = [[]]  # 信息素矩阵
    distance = [[]]  # 距离矩阵
    bestLength = []  # 最佳长度
    bestTour = []  # 最佳路径

    # 三个参数
    alpha = 0.0
    beta = 0.0
    rho = 0.0

    def __init__(self, cityNum=20, antNum=16, MAX_GEN=0, alpha=0.0, beta=0.0, rho=0.0):
        self.cityNum = cityNum
        self.antNum = antNum
        self.ants = [Ant() for i in range(self.antNum)]
        self.MAX_GEN = MAX_GEN
        self.alpha = alpha
        self.beta = beta
        self.rho = rho

    def init(self, filename=None):
        if filename is None:
            return
        X = [0.0 for col in range(self.cityNum)]
        Y = [0.0 for col in range(self.cityNum)]

        for line in open(filename):
            [x, y] = line.split(',')
            X.append(float(x))
            Y.append(float(y))
        # 计算距离矩阵 ，针对具体问题，距离计算方法也不一样，此处用的是att48作为案例，它有48个城市，距离计算方法为伪欧氏距离，最优值为10628
        for i in range(self.cityNum - 1):
            self.distance[i][j] = 0.0
            for j in range(i, self.cityNum):
                rij = sqrt(((x[i] - x[j]) ** 2 + (y[i] - y[j]) ** 2) / 10.0)
                tij = int(rij)
                if tij < rij:
                    self.distance[i][j] = tij + 1
                    self.distance[j][i] = self.distance[i][j]
                else:
                    self.distance[i][j] = tij + 1
                    self.distance[j][i] = self.distance[i][j]
        self.distance[self.cityNum - 1][self.cityNum - 1] = 0.0

        # 初始化信息素矩阵, 初始化为0.1
        self.pheromone = [[0.1 for col in range(self.cityNum)] for row in range(self.cityNum)]
        self.bestLength = float.MAX_VALUE
        self.bestTour = [0 for col in range(self.cityNum + 1)]

        # 随机放置蚂蚁
        for i in range(self.antNum):
            ants[i] = Ant(cityNum=self.cityNum)
            ants[i].init(distance=self.distance, alpha=self.alpha, beta=self.beta)

    def solve(self):
        for g in range(self.MAX_GEN):
            for i in range(self.antNum):
                for j in range(self.cityNum):
                    self.ants[i].selectNextCity(pheromone=self.pheromone)

                self.ants[i].tabu.add(self.ants[i].firstCity)
                if self.ants[i].tourLength < self.bestLength:
                    self.bestLength = self.ants[i].tourLength
                    self.bestTour = self.ants[i].tabu

                for j in range(self.cityNum):
                    self.ants[i].delta[self.ants[i].tabu[j]][self.ants[i].tabu[j + 1]] = 1.0 / self.ants[i].tourLength
                    self.ants[i].delta[self.ants[i].tabu[j + 1]][self.ants[i].tabu[j]] = 1.0 / self.ants[i].tourLength

            # 更新信息素
            self.updatePheromone()

            # 重新初始化蚂蚁
            for i in range(self.antNum):
                self.ants[i].init(distance=self.distance, alpha=self.alpha, beta=self.beta)

    def updatePheromone(self):
        # 信息素挥发
        for i in range(self.cityNum):
            for j in range(self.cityNum):
                self.pheromone[i][j] *= (1 - self.rho)

        # 信息素更新
        for i in range(selfi.cityNum):
            for j in range(self.cityNum):
                for k in range(self.antNum):
                    self.pheromone[i][j] += self.ants[k].delta[i][j]

    def printOptimal(self):
        print 'The optimal length is: %0.2f' % self.bestLength
        print 'The optimal tour is: '
        print self.bestTour


class Usage(Exception):
    def __init__(self, msg):
        self.msg = msg


def main(argv=None):
    if argv is None:
        argv = sys.argv
    try:
        try:
            opts, args = getopt.getopt(argv[1:], "h", ["help"])

            aco = AntColonyOptimization(cityNum=48, antNum=100, MAX_GEN=1000, alpha=1.0, beta=5.0, rho=0.5)
            aco.init("c://data.txt")
            aco.solve()
        except getopt.error, msg:
            raise Usage(msg)
            # more code, unchanged
    except Usage, err:
        print >> sys.stderr, err.msg
        print >> sys.stderr, "for help use --help"
        return 2


if __name__ == "__main__":
    sys.exit(main())

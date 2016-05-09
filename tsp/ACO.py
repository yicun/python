#!/usr/bin/python
# -*- coding: utf-8 -*-
#
# 蚁群算法最早用来求解TSP问题
#

import sys
import getopt
import random
import math
import matplotlib.pyplot as plt


class Ant:
    tabu = []  # 禁忌表
    allowedCities = []  # 允许搜索的城市
    delta = [[]]  # 信息数变化矩阵
    distance = [[]]  # 距离矩阵
    alpha = 0.0
    beta = 0.0

    tourLength = sys.float_info.max  # 路径长度
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
        # 对允许城市列表进行限制, 初始化为所有送餐地点
        self.allowedCities = range(0, self.cityNum, 2)

        # 初始化出发地点, 从允许城市列表中随机选取
        self.firstCity = random.randint(0, self.cityNum - 1) / 2 * 2
        self.allowedCities.remove(self.firstCity)
        self.tabu.append(self.firstCity)
        if self.firstCity % 2 == 0:
            self.allowedCities.append(self.firstCity + 1)
        self.currentCity = self.firstCity

    def selectNextCity(self, pheromone=[[]]):
        if len(self.allowedCities) == 0:
            return

        p = [0 for col in range(self.cityNum)]
        sumPh = 0.0
        # 计算分母部分
        for i in self.allowedCities:
            sumPh += pow(pheromone[self.currentCity][i], self.alpha) * pow(1.0 / self.distance[self.currentCity][i], self.beta)
        # 计算概率矩阵
        for i in range(self.cityNum):
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
        selectCity = -1
        sum1 = 0.0
        for i in range(self.cityNum):
            sum1 += p[i]
            if sum1 >= selectP:
                selectCity = i
                break

        # 从允许选择的城市中去除selectCity
        self.allowedCities.remove(selectCity)
        # 在禁忌表中添加selectCity
        self.tabu.append(selectCity)
        # 修正允许城市列表
        if selectCity % 2 == 0:
            self.allowedCities.append(selectCity + 1)
        # 将当前城市改为选择的城市
        self.currentCity = selectCity

    def fix(self):
        """修正路线, 使其合法"""
        for i in range(len(self.tabu)):
            if self.tabu[i] % 2 == 0 and self.tabu.index(self.tabu[i] + 1) < i:
                idx = self.tabu.index(self.tabu[i] + 1)
                self.tabu[idx] = self.tabu[i]
                self.tabu[i] += 1

    def calculateTourLength(self):
        tourLen = self.distance[self.tabu[0]][self.tabu[0]]
        for i in range(self.cityNum - 1):
            tourLen += self.distance[self.tabu[i]][self.tabu[i + 1]]
        return tourLen

    def getTourLength(self):
        self.tourLength = self.calculateTourLength()
        return self.tourLength


class ACO(Exception):
    ants = []  # 蚂蚁
    antNum = 0  # 蚂蚁数量
    cityNum = 0  # 城市数量
    MAX_GEN = 0  # 运行代数
    pheromone = [[]]  # 信息素矩阵
    distance = [[]]  # 距离矩阵
    bestLength = []  # 最佳长度
    bestTour = []  # 最佳路径

    orderList = [[]]
    cityPositions = [[]]

    # 三个参数
    alpha = 0.0
    beta = 0.0
    rho = 0.0

    def __init__(self, cityNum=20, antNum=16, MAX_GEN=0, alpha=0.0, beta=0.0, rho=0.0, orderList=[[]]):
        self.cityNum = cityNum
        self.antNum = antNum
        self.ants = [Ant() for i in range(self.antNum)]
        self.MAX_GEN = MAX_GEN
        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        self.orderList = orderList

    def init(self, filename=None):
        x = []
        y = []

        self.cityPositions = []
        if self.orderList is not None and len(self.orderList) > 2:
            for idx in range(len(self.orderList)):
                order = self.orderList[idx]
                self.cityPositions.append([idx * 2, order[0], order[1]])
                self.cityPositions.append([idx * 2 + 1, order[3], order[4]])

        if self.cityPositions is not None and len(self.cityPositions) > 2:
            x = map(lambda cp: cp[1], self.cityPositions)
            y = map(lambda cp: cp[2], self.cityPositions)
        elif filename is not None:
            # 从文件读取坐标位置，格式: cityId X Y
            for line in open(filename):
                [id, xx, yy] = line.split(' ')
                x.append(float(xx))
                y.append(float(yy))
        else:
            print '## City position info is miss!'

        # 计算距离矩阵 ，针对具体问题，距离计算方法也不一样，此处用的是att48作为案例，它有48个城市，距离计算方法为伪欧氏距离，最优值为10628
        self.distance = [[0.0 for col in range(self.cityNum)] for row in range(self.cityNum)]
        for i in range(self.cityNum - 1):
            self.distance[i][i] = math.sqrt((x[i] - 57) ** 2 + (y[i] - 34) ** 2)
            for j in range(i + 1, self.cityNum):
                self.distance[i][j] = math.sqrt((x[i] - x[j]) ** 2 + (y[i] - y[j]) ** 2)
                self.distance[j][i] = self.distance[i][j]
        self.distance[self.cityNum - 1][self.cityNum - 1] = math.sqrt((x[self.cityNum - 1] - 57) ** 2 + (y[self.cityNum - 1] - 34) ** 2)

        # 初始化信息素矩阵, 初始化为0.1
        self.pheromone = [[0.1 for col in range(self.cityNum)] for row in range(self.cityNum)]
        self.bestLength = sys.float_info.max
        self.bestTour = [-1 for col in range(self.cityNum)]

        # 随机放置蚂蚁
        for i in range(self.antNum):
            self.ants[i] = Ant(cityNum=self.cityNum)
            self.ants[i].init(distance=self.distance, alpha=self.alpha, beta=self.beta)

    def solve(self):
        for g in range(self.MAX_GEN):
            print '## Start generation: %d' % g
            for i in range(self.antNum):
                for j in range(self.cityNum - 1):
                    self.ants[i].selectNextCity(pheromone=self.pheromone)
                # self.ants[i].tabu.append(self.ants[i].firstCity)
                # self.ants[i].fix()
                if self.ants[i].getTourLength() < self.bestLength:
                    self.bestLength = self.ants[i].getTourLength()
                    self.bestTour = self.ants[i].tabu[:]

                for j in range(self.cityNum - 1):
                    self.ants[i].delta[self.ants[i].tabu[j]][self.ants[i].tabu[j + 1]] = 1.0 / self.ants[i].getTourLength()
                    self.ants[i].delta[self.ants[i].tabu[j + 1]][self.ants[i].tabu[j]] = 1.0 / self.ants[i].getTourLength()

            # 更新信息素
            self.updatePheromone()

            # 重新初始化蚂蚁
            for i in range(self.antNum):
                self.ants[i].init(distance=self.distance, alpha=self.alpha, beta=self.beta)

            # 打印信息
            self.printOptimal()

        self.drawTour()

    def drawTour(self):
        x = map(lambda t: self.cityPositions[t][1], self.bestTour)
        y = map(lambda t: self.cityPositions[t][2], self.bestTour)
        shop_x = map(lambda t: self.cityPositions[t][1], filter(lambda t: t % 2 == 0, self.bestTour))
        user_x = map(lambda t: self.cityPositions[t][1], filter(lambda t: t % 2 == 1, self.bestTour))

        shop_y = map(lambda t: self.cityPositions[t][2], filter(lambda t: t % 2 == 0, self.bestTour))
        user_y = map(lambda t: self.cityPositions[t][2], filter(lambda t: t % 2 == 1, self.bestTour))

        x.insert(0, 57)
        y.insert(0, 34)
        print x
        print y
        plt.figure(1)
        # plt.plot(shop_x, shop_y, 'bo', user_x, user_y, 'yo', x, y)
        plt.plot(shop_x, shop_y, 'bo', user_x, user_y, 'yo', x, y)
        for i in range(self.cityNum):
            if i % 2 == 0:
                plt.annotate('s' + str(i), xy=(self.cityPositions[i][1], self.cityPositions[i][2]),
                             xytext=(self.cityPositions[i][1] + 0.5, self.cityPositions[i][2]))
            else:
                plt.annotate('u' + str(i), xy=(self.cityPositions[i][1], self.cityPositions[i][2]),
                             xytext=(self.cityPositions[i][1] + 0.5, self.cityPositions[i][2]))

        # plt.plot(x, y, 'bo')
        plt.show()

    def updatePheromone(self):
        # 信息素挥发
        for i in range(self.cityNum):
            for j in range(self.cityNum):
                self.pheromone[i][j] *= (1 - self.rho)

        # 信息素更新
        for i in range(self.cityNum):
            for j in range(self.cityNum):
                for k in range(self.antNum):
                    self.pheromone[i][j] += self.ants[k].delta[i][j]

    def printOptimal(self):
        print 'The optimal length is: %0.2f' % self.bestLength
        print 'The optimal tour is: ', self.bestTour


class Usage(Exception):
    def __init__(self, msg):
        self.msg = msg


def main(argv=None):
    if argv is None:
        argv = sys.argv
    try:
        try:
            opts, args = getopt.getopt(argv[1:], "h", ["help"])
            order_list = [
                [31, 54, -2, 49, 47, 37],
                [21, 17, 8, 31, 50, 36],
                [11, 36, 5, 52, 27, 30],
                [16, 8, 6, 7, 27, 38],
                [52, 58, 9, 37, 38, 35],
                [23, 25, 6, 48, 56, 40],
                [32, 47, 11, 63, 62, 54],
                [43, 69, 14, 22, 35, 46],
                [29, 32, 15, 55, 28, 54],
                [20, 47, 13, 43, 62, 39]]

            aco = ACO(cityNum=len(order_list) * 2, antNum=50, MAX_GEN=50, alpha=1.0, beta=5.0, rho=0.5, orderList=order_list)
            aco.init()
            aco.solve()

            # lengths = []
            # tours = []
            # for i in range(100):
            #     aco = ACO(cityNum=len(order_list) * 2, antNum=50, MAX_GEN=100, alpha=1.0, beta=5.0, rho=0.5, orderList=order_list)
            #     aco.init()
            #     aco.solve()
            #     print "The best length: %.2f" % aco.bestLength
            #     print "The best tour: ", aco.bestTour
            #     lengths.append(aco.bestLength)
            #     tours.append(aco.bestTour)
            # print lengths
            # print tours

        except getopt.error, msg:
            raise Usage(msg)
            # more code, unchanged
    except Usage, err:
        print >> sys.stderr, err.msg
        print >> sys.stderr, "for help use --help"
        return 2


if __name__ == "__main__":
    sys.exit(main())

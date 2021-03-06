#!/usr/bin/python
# -*- coding: utf-8 -*-
#
# 利用遗传算法求解TSP问题
#

import sys
import getopt
import random
import math
import copy
import matplotlib.pyplot as plt


class Chromosome:
    tour = []
    distance = [[]]
    cityNum = 0
    fitness = 0.0

    def __init__(self, cityNum=0, distance=None, tour=None, fitness=0.0):
        self.cityNum = cityNum
        if distance is not None:
            self.distance = copy.deepcopy(distance)
        else:
            self.distance = []
        if tour is not None:
            self.tour = copy.deepcopy(tour)
        self.fitness = fitness

    def randomGeneration(self):
        allowedCities = range(self.cityNum)
        self.tour = [-1 for col in range(self.cityNum)]
        for i in range(self.cityNum):
            index = random.randint(0, len(allowedCities) - 1)
            selectedCity = allowedCities[index]
            self.tour[i] = selectedCity
            allowedCities.remove(selectedCity)
        self.calculateFitness()

    def calculateFitness(self):
        self.fitness = 0.0
        total_len = self.distance[self.tour[0]][self.tour[0]]
        for i in range(self.cityNum - 1):
            total_len += self.distance[self.tour[i]][self.tour[i + 1]]
        # total_len += self.distance[0][self.tour[self.cityNum - 1]]
        self.fitness = 1.0 / total_len

        return self.fitness

    def getFitness(self):
        self.fitness = self.calculateFitness()
        return self.fitness

    def clone(self):
        self.calculateFitness()
        return Chromosome(cityNum=self.cityNum, distance=self.distance, tour=self.tour, fitness=self.fitness)

    def printInfo(self):
        print self.tour
        print 'fitness measure is: %.5f' % self.fitness


class GA(Exception):
    chromosomes = []
    nextGeneration = []
    N = 100
    cityNum = 30
    p_c_t = 0.9
    p_m_t = 0.1
    MAX_GEN = 1000
    bestLength = sys.float_info.max
    bestTour = []
    bestFitness = 0.0
    bestGeneration = 0
    averageFitness = 0.0
    distance = [[]]
    cityPositions = [[]]
    orderList = [[]]
    filename = ''

    def __init__(self, N=100, cityNum=30, MAX_GEN=1000, p_c_t=0.9, p_m_t=0.1, cityPositions=[[]], orderList=[[]], filename=None):
        self.N = N
        self.cityNum = cityNum
        self.MAX_GEN = MAX_GEN
        self.p_c_t = p_c_t
        self.p_m_t = p_m_t
        self.bestTour = [-1 for row in range(self.cityNum)]
        self.averageFitness = [0.0 for row in range(self.MAX_GEN)]
        self.bestFitness = 0.0
        self.chromosomes = []
        self.distance = [[sys.float_info.max for col in range(self.cityNum)] for row in range(self.cityNum)]
        self.cityPositions = cityPositions
        self.orderList = orderList
        self.filename = filename

    def solve(self):
        print '## Start initialization:'
        self.init()
        print '## End initialization!'
        print '## Start evolution:'
        for i in range(self.MAX_GEN):
            print '## Start generation: %d' % i
            self.evolve(i)
            print '## End generation: %d' % i
            # break
        print '## End evolution!'

        self.printOptimal()
        self.drawTour()
        # self.outputResults()

    def init(self):
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
        elif self.filename is not None:
            # 从文件读取坐标位置，格式: cityId X Y
            for line in open(self.filename):
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

        for i in range(self.N):
            chromosome = Chromosome(cityNum=self.cityNum, distance=self.distance)
            chromosome.randomGeneration()
            self.chromosomes.append(chromosome)
            chromosome.printInfo()

    def rank(self):
        for i in range(self.cityNum):
            self.chromosomes[i].calculateFitness()
        for i in range(self.cityNum):
            for j in range(self.cityNum):
                if self.chromosomes[i].fitness > self.chromosomes[j].fitness:
                    tmp = self.chromosomes[i]
                    self.chromosomes[i] = self.chromosomes[j]
                    self.chromosomes[j] = tmp

    @staticmethod
    def PMX(tour1=[], tour2=[]):
        """部分匹配法（Partially Matching Crossover， PMX）"""
        # 定义两个cut点
        tour_length = len(tour1)
        cutPoint1 = random.randint(0, tour_length - 2)
        cutPoint2 = random.randint(cutPoint1 + 1, tour_length - 1)

        cut_tour1 = tour1[cutPoint1: cutPoint2 + 1]
        cut_tour2 = tour2[cutPoint1: cutPoint2 + 1]

        point_map = [dict(), dict()]
        for j in range(cutPoint2 - cutPoint1 + 1):
            point_map[0][cut_tour1[j]] = cut_tour2[j]
            point_map[1][cut_tour2[j]] = cut_tour1[j]

        tour1[cutPoint1: cutPoint2 + 1] = cut_tour2
        tour2[cutPoint1: cutPoint2 + 1] = cut_tour1

        for j in range(tour_length):
            if j < cutPoint1 or j > cutPoint2:
                while tour1[j] in cut_tour2:
                    tour1[j] = point_map[1][tour1[j]]

                while tour2[j] in cut_tour1:
                    tour2[j] = point_map[0][tour2[j]]
        return [tour1, tour2]

    @staticmethod
    def CX(tour1=[], tour2=[]):
        """循环交叉法（Cycle Crossover， CX）"""
        tmp_tour1 = [-1 for col in range(len(tour1))]
        idx = 0
        tmp_tour1[idx] = tour1[idx]
        while True:
            idx = tour1.index(tour2[idx])
            tmp_tour1[idx] = tour1[idx]
            if tour2[idx] == tmp_tour1[0]:
                break
        for j in range(len(tmp_tour1)):
            if tmp_tour1[j] == -1:
                tmp_tour1[j] = tour2[j]

        tmp_tour2 = [-1 for col in range(len(tour2))]
        idx = 0
        tmp_tour2[idx] = tour2[idx]
        while True:
            idx = tour2.index(tour1[idx])
            tmp_tour2[idx] = tour2[idx]
            if tour1[idx] == tmp_tour2[0]:
                break
        for j in range(len(tmp_tour2)):
            if tmp_tour2[j] == -1:
                tmp_tour2[j] = tour1[j]

        return [tmp_tour1, tmp_tour2]

    @staticmethod
    def OX1(tour1=[], tour2=[]):
        """次序交叉法1（Order Crossover， OX1）"""
        # 定义两个cut点
        tour_length = len(tour1)
        cutPoint1 = random.randint(0, tour_length - 2)
        cutPoint2 = random.randint(cutPoint1 + 1, tour_length - 1)

        cut_tour1 = tour1[cutPoint1: cutPoint2 + 1]
        cut_tour2 = tour2[cutPoint1: cutPoint2 + 1]

        tmp = tour1[cutPoint2 + 1:tour_length]
        tmp.extend(tour1[0:cutPoint2 + 1])
        remain_tour1 = filter(lambda x: x not in cut_tour2, tmp)
        tmp = tour2[cutPoint2 + 1:tour_length]
        tmp.extend(tour2[0:cutPoint2 + 1])
        remain_tour2 = filter(lambda x: x not in cut_tour1, tmp)

        tour1 = remain_tour1[len(remain_tour1) - cutPoint1:len(remain_tour1)]
        tour1.extend(cut_tour2)
        tour1.extend(remain_tour1[0:tour_length - cutPoint2 - 1])
        tour2 = remain_tour2[len(remain_tour2) - cutPoint1:len(remain_tour2)]
        tour2.extend(cut_tour1)
        tour2.extend(remain_tour2[0:tour_length - cutPoint2 - 1])

        return [tour1, tour2]

    @staticmethod
    def OX2(tour1=[], tour2=[]):
        """ 次序交叉法2（Order Crossover， OX2）"""
        # 定义三个cut点
        tour_length = len(tour1)
        cutPoint1 = random.randint(0, tour_length - 3)
        cutPoint2 = random.randint(cutPoint1 + 1, tour_length - 2)
        cutPoint3 = random.randint(cutPoint2 + 1, tour_length - 1)

        cut_tour1 = tour1[cutPoint1: cutPoint2 + 1]
        cut_tour1.append(tour1[cutPoint3])
        cut_tour1.reverse()
        cut_tour2 = tour2[cutPoint1: cutPoint2 + 1]
        cut_tour2.append(tour2[cutPoint3])
        cut_tour2.reverse()

        remain_tour1 = filter(lambda x: x not in cut_tour2, tour1)
        remain_tour1.reverse()
        remain_tour2 = filter(lambda x: x not in cut_tour1, tour2)
        remain_tour2.reverse()

        for i in range(tour_length):
            if tour1[i] not in remain_tour1:
                tour1[i] = cut_tour2.pop()
            if tour2[i] not in remain_tour2:
                tour2[i] = cut_tour1.pop()

        return [tour1, tour2]

    @staticmethod
    def POS(tour1=[], tour2=[]):
        """基于位置的交叉法（Position Based Crossover， POS）"""
        # 定义三个cut点
        tour_length = len(tour1)
        cutPoint1 = random.randint(0, tour_length - 3)
        cutPoint2 = random.randint(cutPoint1 + 1, tour_length - 2)
        cutPoint3 = random.randint(cutPoint2 + 1, tour_length - 1)

        cut_tour1 = tour1[cutPoint1: cutPoint2 + 1]
        cut_tour1.append(tour1[cutPoint3])
        cut_tour1.reverse()
        cut_tour2 = tour2[cutPoint1: cutPoint2 + 1]
        cut_tour2.append(tour2[cutPoint3])
        cut_tour2.reverse()

        remain_tour1 = filter(lambda x: x not in cut_tour2, tour1)
        remain_tour1.reverse()
        remain_tour2 = filter(lambda x: x not in cut_tour1, tour2)
        remain_tour2.reverse()

        for i in range(tour_length):
            if cutPoint1 <= i <= cutPoint2 or i == cutPoint3:
                tour1[i] = cut_tour2.pop()
                tour2[i] = cut_tour1.pop()
            else:
                tour1[i] = remain_tour1.pop()
                tour2[i] = remain_tour2.pop()

        return [tour1, tour2]

    @staticmethod
    def APX(tour1=[], tour2=[]):
        """ 交替位置交叉法（Alternating Position Crossover，APX）"""
        tmp_tour1 = []
        tmp_tour2 = []
        tour1.reverse()
        tour2.reverse()
        for i in range(len(tour1)):
            t1 = tour1.pop()
            t2 = tour2.pop()
            if t1 not in tmp_tour1:
                tmp_tour1.append(t1)
            if t2 not in tmp_tour1:
                tmp_tour1.append(t2)
            if t2 not in tmp_tour2:
                tmp_tour2.append(t2)
            if t1 not in tmp_tour2:
                tmp_tour2.append(t1)
        return [tmp_tour1, tmp_tour2]

    @staticmethod
    def DM(tour=[]):
        """替换变异（Displacement Mutation， DM）"""
        # 定义两个cut点
        tour_length = len(tour)
        cutPoint1 = random.randint(0, tour_length - 2)
        cutPoint2 = random.randint(cutPoint1 + 1, tour_length - 1)

        cut_tour = tour[cutPoint1: cutPoint2 + 1]
        remain_tour = filter(lambda x: x not in cut_tour, tour)

        position = random.randint(0, max(len(remain_tour) - 1, 0))
        tour = remain_tour[0: position]
        tour.extend(cut_tour)
        tour.extend(remain_tour[position:len(remain_tour)])

        return tour

    @staticmethod
    def EM(tour=[]):
        """交换变异（Exchange Mutation， EM）"""
        # 定义两个cut点
        tour_length = len(tour)
        cutPoint1 = random.randint(0, tour_length - 2)
        cutPoint2 = random.randint(cutPoint1 + 1, tour_length - 1)
        tmp = tour[cutPoint1]
        tour[cutPoint1] = tour[cutPoint2]
        tour[cutPoint2] = tmp
        return tour

    @staticmethod
    def EM2(tour=[]):
        """单点交换变异（Exchange Mutation， EM）"""
        # 定义1个cut点
        tour_length = len(tour)
        cutPoint = random.randint(0, tour_length - 2)
        tmp = tour[cutPoint]
        tour[cutPoint] = tour[cutPoint + 1]
        tour[cutPoint + 1] = tmp
        return tour

    @staticmethod
    def IM(tour=[]):
        """ 插入变异（Insertion Mutation， IM）"""
        cutPoint = random.randint(0, len(tour) - 1)
        point = tour[cutPoint]
        tour.remove(point)
        position = random.randint(0, len(tour))
        tour.insert(position, point)
        return tour

    @staticmethod
    def SIM(tour=[]):
        """替换变异（Displacement Mutation， DM）"""
        # 定义两个cut点
        tour_length = len(tour)
        cutPoint1 = random.randint(0, tour_length - 2)
        cutPoint2 = random.randint(cutPoint1 + 1, tour_length - 1)
        for i in range(0, (cutPoint2 - cutPoint1 + 1) / 2):
            tmp = tour[cutPoint1 + i]
            tour[cutPoint1 + i] = tour[cutPoint2 - i]
            tour[cutPoint2 - i] = tmp
        return tour

    @staticmethod
    def IVM(tour=[]):
        """倒位变异（Inversion Mutation， IVM）"""
        # 定义两个cut点
        tour_length = len(tour)
        cutPoint1 = random.randint(0, tour_length - 2)
        cutPoint2 = random.randint(cutPoint1 + 1, tour_length - 1)

        cut_tour = tour[cutPoint1:cutPoint2 + 1]
        cut_tour.reverse()
        remain_tour = filter(lambda x: x not in cut_tour, tour)

        position = random.randint(0, max(len(remain_tour) - 1, 0))
        tour = remain_tour[0:position]
        tour.extend(cut_tour)
        tour.extend(remain_tour[position:len(remain_tour)])
        return tour

    @staticmethod
    def SM(tour=[]):
        """争夺变异（Scramble Mutation， SM）"""
        # 定义两个cut点
        tour_length = len(tour)
        cutPoint1 = random.randint(0, tour_length - 2)
        cutPoint2 = random.randint(cutPoint1 + 1, tour_length - 1)

        cut_tour = tour[cutPoint1:cutPoint2 + 1]
        point = cut_tour[0]
        cut_tour.remove(point)
        cut_tour.append(point)

        remain_tour = filter(lambda x: x not in cut_tour, tour)

        position = random.randint(0, max(len(remain_tour) - 1, 0))
        tour = remain_tour[0:position]
        tour.extend(cut_tour)
        tour.extend(remain_tour[position:len(remain_tour)])
        return tour

    @staticmethod
    def FIX(tour=[]):
        """修正路线, 使其合法"""
        for i in range(len(tour)):
            if tour[i] % 2 == 0 and tour.index(tour[i] + 1) < i:
                idx = tour.index(tour[i] + 1)
                tour[idx] = tour[i]
                tour[i] += 1
        return tour

    @staticmethod
    def OPT(tour=[], p_o_t=0.3, distance=[[]]):
        """路线优化, optimize"""

        # 四点原则
        if distance[tour[0]][tour[0]] + distance[tour[1]][tour[2]] > distance[tour[1]][tour[1]] + distance[tour[0]][tour[2]]:
            tmp = tour[0]
            tour[0] = tour[1]
            tour[1] = tmp
        for i in range(len(tour) - 4):
            if random.random() < p_o_t and distance[tour[i]][tour[i + 1]] + distance[tour[i + 2]][tour[i + 3]] > distance[tour[i]][tour[i + 2]] + \
                    distance[tour[i + 1]][tour[i + 3]]:
                tmp = tour[i + 1]
                tour[i + 1] = tour[i + 2]
                tour[i + 2] = tmp

        if distance[tour[0]][tour[0]] < distance[tour[3]][tour[3]]:
            tmp = tour[0]
            tour[0] = tour[3]
            tour[3] = tmp
        for i in range(len(tour) - 4):
            if random.random() < p_o_t and distance[tour[i]][tour[i + 1]] > distance[tour[i]][tour[i + 3]]:
                tmp = tour[i + 1]
                tour[i + 1] = tour[i + 3]
                tour[i + 3] = tmp

        # 三点原则
        if distance[tour[0]][tour[0]] > distance[tour[1]][tour[1]]:
            tmp = tour[0]
            tour[0] = tour[1]
            tour[1] = tmp
        for i in range(len(tour) - 3):
            if random.random() < p_o_t and distance[tour[i]][tour[i + 1]] > distance[tour[i]][tour[i + 2]]:
                tmp = tour[i + 1]
                tour[i + 1] = tour[i + 2]
                tour[i + 2] = tmp

        return tour

    def evolve(self, g=0):
        self.chromosomes.sort(lambda c1, c2: cmp(c2.getFitness, c1.getFitness))
        # self.rank()

        selectionP = [0.0 for row in range(self.N / 2)]
        total_fitness = 0.0
        # 舍弃一半后代
        for i in range(self.N / 2):
            total_fitness += self.chromosomes[i].calculateFitness()
            # print self.chromosomes[i].tour, '\t', self.chromosomes[i].fitness

            # 计算当前最佳值
            if self.chromosomes[i].fitness > self.bestFitness:
                self.bestGeneration = g - 1
                self.bestFitness = self.chromosomes[i].fitness
                self.bestLength = 1.0 / self.bestFitness
                self.bestTour = copy.deepcopy(self.chromosomes[i].tour)
        self.averageFitness[g] = total_fitness / self.N

        print '## The average fitness in %d generation is : %.5f, and the best fitness is: %.5f' % (g, self.averageFitness[g], self.bestFitness)
        for i in range(self.N / 2):
            if i == 0:
                selectionP[i] = self.chromosomes[i].fitness / total_fitness
            else:
                selectionP[i] = selectionP[i - 1] + self.chromosomes[i].fitness / total_fitness

        self.nextGeneration = []
        while len(self.nextGeneration) < self.N:
            children = []
            # 轮盘赌选择两个染色体
            # print '## Start selection:'
            for j in [0, 1]:
                selectionCity = -1
                p = random.random()
                for k in range(self.N - 1):
                    if k == 0 and p < selectionP[k]:
                        selectionCity = 0
                        break
                    if selectionP[k] < p <= selectionP[k + 1]:
                        selectionCity = k + 1
                        break

                children.append(self.chromosomes[selectionCity].clone())

            # 交叉操作
            # print '## Start crossover:'
            if random.random() < self.p_c_t:
                """次序交叉法1（Order Crossover， OX1）"""
                [children[0].tour, children[1].tour] = self.PMX(tour1=children[0].tour, tour2=children[1].tour)
                [children[0].tour, children[1].tour] = self.CX(tour1=children[0].tour, tour2=children[1].tour)
                [children[0].tour, children[1].tour] = self.OX1(tour1=children[0].tour, tour2=children[1].tour)
                [children[0].tour, children[1].tour] = self.OX2(tour1=children[0].tour, tour2=children[1].tour)
                [children[0].tour, children[1].tour] = self.POS(tour1=children[0].tour, tour2=children[1].tour)
                [children[0].tour, children[1].tour] = self.APX(tour1=children[0].tour, tour2=children[1].tour)

                for j in [0, 1]:
                    self.nextGeneration.append(children[j].clone())

            # 变异操作(DM)
            # print '## Start mutation:'
            if random.random() < self.p_m_t:
                """替换变异（Displacement Mutation， DM）"""
                for j in [0, 1]:
                    tour = children[j].tour
                    # tour = self.DM(tour=tour)
                    # tour = self.EM(tour=tour)
                    tour = self.EM2(tour=tour)
                    # tour = self.IM(tour=tour)
                    # tour = self.SIM(tour=tour)
                    # tour = self.IVM(tour=tour)
                    # tour = self.SM(tour=tour)
                    children[j].tour = tour
                    self.nextGeneration.append(children[j])

        for k in range(self.N):
            chromosome = self.nextGeneration[k].clone()
            # 路线规则优化(OPT)
            chromosome.tour = self.OPT(tour=chromosome.tour, p_o_t=1.1, distance=self.distance)
            # 路线修正(FIX)
            chromosome.tour = self.FIX(chromosome.tour)
            chromosome.tour = self.OPT(tour=chromosome.tour, p_o_t=1.1, distance=self.distance)
            # 路线修正(FIX)
            chromosome.tour = self.FIX(chromosome.tour)
            chromosome.tour = self.OPT(tour=chromosome.tour, p_o_t=1.1, distance=self.distance)
            # 路线修正(FIX)
            chromosome.tour = self.FIX(chromosome.tour)
            chromosome.tour = self.OPT(tour=chromosome.tour, p_o_t=1.1, distance=self.distance)
            # 路线修正(FIX)
            chromosome.tour = self.FIX(chromosome.tour)
            chromosome.tour = self.OPT(tour=chromosome.tour, p_o_t=1.1, distance=self.distance)
            # 路线修正(FIX)
            chromosome.tour = self.FIX(chromosome.tour)
            self.chromosomes[k] = chromosome.clone()

    def printOptimal(self):
        print 'The best generation is: %d' % self.bestGeneration
        print 'The best fitness is: %.5f' % self.bestFitness
        print 'The best tour length is: %.5f' % self.bestLength
        print 'The best tour is: '
        print self.bestTour

    def outputResults(self):
        self.printOptimal()

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
            plt.annotate(str(i), xy=(self.cityPositions[i][1], self.cityPositions[i][2]), xytext=(self.cityPositions[i][1], self.cityPositions[i][2]))
        # plt.plot(x, y, 'bo')
        plt.show()


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

            ga = GA(N=100, cityNum=len(order_list) * 2, MAX_GEN=500, p_c_t=0.4, p_m_t=1.0, orderList=order_list)
            ga.solve()
            # lengths = []
            # for k in range(100):
            #     ga = GA(N=50, cityNum=len(order_list) * 2, MAX_GEN=500, p_c_t=0.4, p_m_t=1.0, orderList=order_list)
            #     ga.solve()
            #     lengths.append(ga.bestLength)
            #
            # print lengths

            # tour = GA.DM(tour=[1, 2, 3, 4, 5, 6, 7, 8])
            # print tour
            # tour = GA.FIX(tour=[9, 4, 6, 7, 5, 3, 0, 2, 8, 1])
            # print tour
            # tour1 = [1, 2, 3, 4, 5, 6, 7, 8]
            # tour2 = [2, 4, 6, 8, 7, 5, 3, 1]
            # print tour1
            # print tour2
            # [tour1, tour2] = GA.PMX(tour1=[1, 2, 3, 4, 5, 6, 7, 8], tour2=[2, 4, 6, 8, 7, 5, 3, 1])
            # print tour1
            # print tour2
            # print
            # [tour1, tour2] = GA.CX(tour1=[1, 2, 3, 4, 5, 6, 7, 8], tour2=[2, 4, 6, 8, 7, 5, 3, 1])
            # print tour1
            # print tour2
            # print
            # [tour1, tour2] = GA.OX1(tour1=[1, 2, 3, 4, 5, 6, 7, 8], tour2=[2, 4, 6, 8, 7, 5, 3, 1])
            # print tour1
            # print tour2
            # print
            # [tour1, tour2] = GA.OX2(tour1=[1, 2, 3, 4, 5, 6, 7, 8], tour2=[2, 4, 6, 8, 7, 5, 3, 1])
            # print tour1
            # print tour2
            # print
            # [tour1, tour2] = GA.POS(tour1=[1, 2, 3, 4, 5, 6, 7, 8], tour2=[2, 4, 6, 8, 7, 5, 3, 1])
            # print tour1
            # print tour2
            # print
            # [tour1, tour2] = GA.APX(tour1=[1, 2, 3, 4, 5, 6, 7, 8], tour2=[2, 4, 6, 8, 7, 5, 3, 1])
            # print tour1
            # print tour2
            # print
            # tour = [1, 2, 3, 4, 5, 6, 7, 8]
            # print tour
            # tour = GA.DM(tour=[1, 2, 3, 4, 5, 6, 7, 8])
            # print tour
            # tour = GA.EM(tour=[1, 2, 3, 4, 5, 6, 7, 8])
            # print tour
            # tour = GA.IM(tour=[1, 2, 3, 4, 5, 6, 7, 8])
            # print tour
            # tour = GA.SIM(tour=[1, 2, 3, 4, 5, 6, 7, 8])
            # print tour
            # tour = GA.IVM(tour=[1, 2, 3, 4, 5, 6, 7, 8])
            # print tour
            # tour = GA.SM(tour=[1, 2, 3, 4, 5, 6, 7, 8])
            # print tour
        except getopt.error, msg:
            raise Usage(msg)
            # more code, unchanged
    except Usage, err:
        print >> sys.stderr, err.msg
        print >> sys.stderr, "for help use --help"
        return 2


if __name__ == "__main__":
    sys.exit(main())

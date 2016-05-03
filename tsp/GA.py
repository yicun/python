#!/usr/bin/python
# -*- coding: utf-8 -*-
#
# 利用遗传算法求解TSP问题
#

import sys
import getopt
import random
import math


class Chromosome:
    tour = []
    distance = [[]]
    cityNum = 0
    fitness = 0.0

    def __init__(self, cityNum=0, distance=[[]], tour=[], fitness=0.0):
        self.cityNum = cityNum
        self.distance = distance
        self.tour = [-1 for row in range(self.cityNum)]

    def randomGeneration(self):
        allowedCities = range(self.cityNum)

        for i in range(self.cityNum):
            index = random.randint(len(allowedCities))
            selectedCity = allowedCities[index]
            self.tour[i] = selectedCity
            allowedCities.remove(selectedCity)

    def calculateFitness(self):
        self.fitness = 0.0;
        total_len = 0.0
        for i in range(self.cityNum - 1):
            total_len += self.distance[self.tour[i]][self.tour[i + 1]]
        total_len += self.distance[0][self.tour[self.cityNum - 1]]
        self.fitness = 1.0 / total_len

        return self.fitness

    def getFitness(self):
        self.fitness = self.calculateFitness()
        return self.fitness

    def clone(self):
        self.calculateFitness
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
    averageFitness = 0.0
    distance = [[]]
    filename = ''

    def __init__(self, N=100, cityNum=30, MAX_GEN=1000, p_c_t=0.9, p_m_t=0.1, filename=None):
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
        self.filename = filename

    def solve(self):
        print '## Start initialization:'
        self.init();
        print '## End initialization!'
        print '## Start evolution:'
        for i in range(self.MAX_GEN):
            print '## Start generation: %d' % i
            self.evolve(i)
            print '## End generation: %d' % i
        print '## End evolution!'

        self.printOptimal()
        self.outputResults()

    def init(self):
        x = []
        y = []
        # 从文件读取坐标位置，格式: cityId X Y
        for line in open(filename):
            [id, xx, yy] = line.split(' ')
            x.append(float(xx))
            y.append(float(yy))

        # 计算距离矩阵 ，针对具体问题，距离计算方法也不一样，此处用的是att48作为案例，它有48个城市，距离计算方法为伪欧氏距离，最优值为10628
        self.distance = [[0.0 for col in range(self.cityNum)] for row in range(self.cityNum)]
        for i in range(self.cityNum - 1):
            self.distance[i][i] = 0.0
            for j in range(i, self.cityNum):
                rij = math.sqrt(((x[i] - x[j]) ** 2 + (y[i] - y[j]) ** 2) / 10.0)
                tij = int(rij)
                # if tij < rij:
                self.distance[i][j] = tij + 1
                self.distance[j][i] = self.distance[i][j]
                # else:
                # self.distance[i][j] = tij + 1
                # self.distance[j][i] = self.distance[i][j]
        self.distance[self.cityNum - 1][self.cityNum - 1] = 0.0

        for i in range(self.N):
            chromosome = Chromosome(cityNum=self.cityNum, distance=self.distance)
            chromosome.randomGeneration()
            self.chromosomes.append(chromosome)
            chromosome.printInfo()

    @staticmethod
    def OX1(tour1=[], tour2=[]):
        """次序交叉法1（Order Crossover， OX1）"""
        # 定义两个cut点
        tour_length = len(tour1)
        cutPoint1 = random.randint(0, tour_length - 2)
        cutPoint2 = random.randint(cutPoint1 + 1, tour_length - 1)
        print '## Cut point1: %d and point2: %d' % (cutPoint1, cutPoint2)

        cut_tour1 = tour1[cutPoint1: cutPoint2 + 1]
        cut_tour2 = tour2[cutPoint1: cutPoint2 + 1]

        remain_tour1 = filter(lambda x: x not in cut_tour2, tour1)
        remain_tour2 = filter(lambda x: x not in cut_tour1, tour2)

        tour1 = remain_tour1[len(remain_tour1) - cutPoint1:len(remain_tour1)]
        tour1.extend(cut_tour2)
        tour1.extend(remain_tour1[0:tour_length - cutPoint2 - 1])
        tour2 = remain_tour2[len(remain_tour2) - cutPoint1:len(remain_tour2)]
        tour2.extend(cut_tour1)
        tour2.extend(remain_tour2[0:tour_length - cutPoint2 - 1])

        return [tour1, tour2]

    @staticmethod
    def DM(tour=[]):
        """替换变异（Displacement Mutation， DM）"""
        # 定义两个cut点
        tour_length = len(tour)
        cutPoint1 = random.randint(0, tour_length - 2)
        cutPoint2 = random.randint(cutPoint1 + 1, tour_length - 1)
        print '## Cut point1: %d and point2: %d' % (cutPoint1, cutPoint2)

        cut_tour = tour[cutPoint1: cutPoint2 + 1]
        print '## Cut tour: '
        print cut_tour

        remain_tour = filter(lambda x: x not in cut_tour, tour)

        position = random.randint(0, len(remain_tour) - 1)
        print '## Insert position: %d' % position
        tour = remain_tour[0: position]
        tour.extend(cut_tour)
        tour.extend(remain_tour[position:len(remain_tour)])

        return tour

    def evolve(self, g=0):
        selectionP = [0.0 for row in range(self.N)]
        total_fitness = 0.0
        for i in range(self.N):
            total_fitness += self.chromosomes[i].calculateFitness()
            if self.chromosomes[i].fitness > self.bestFitness:
                self.bestFitness = self.chromosomes[i].fitness
                self.bestLength = 1.0 / self.bestFitness
                self.bestTour = self.chromosomes[i].tour
        self.averageFitness[g] = total_fitness / self.N

        print '## The average fitness in %d generation is : %.5f, and the best fitness is: %.5f' % (g, self.averageFitness[g], self.bestFitness)
        for i in range(self.N):
            if i == 0:
                selectionP[i] = self.chromosomes[i].fitness / total_fitness
            else:
                selectionP[i] = selectionP[i - 1] + self.chromosomes[i].fitness / total_fitness

        for i in range(0, self.N, 2):
            children = []
            # 轮盘赌选择两个染色体
            print '## Start selection:'
            for j in [0, 1]:
                selectionCity = 0
                for k in range(self.N - 1):
                    p = random.random()
                    if selectionP[k] < p <= selectionP[k + 1]:
                        selectionCity = k
                    if k == 0 and random.random() < selectionP[k]:
                        selectionCity = 0

                children.append(self.chromosomes[selectionCity].clone())

            # 交叉操作
            print '## Start crossover:'
            if random.random < self.p_c_t:
                """次序交叉法1（Order Crossover， OX1）"""
                [tour1, tour2] = self.OX1(tour1=children[0].tour, tour2=children[1].tour)
                children[0].tour = tour1
                children[1].tour = tour2

            # 变异操作(DM)
            print '## Start mutation:'
            if random.random() < self.p_m_t:
                for j in [0, 1]:
                    children[j].tour = self.DM(tour=children[j].tour)
                    # # 定义两个cut点
                    # cutPoint1 = -1
                    # cutPoint2 = -1
                    # r1 = random.randint(self.cityNum - 1)
                    # if 0 < r1 < self.cityNum - 1:
                    #     cutPoint1 = r1
                    #     cutPoint2 = random.randint(r1 + 1, self.cityNum - 1)
                    #
                    # if cutPoint1 > 0 and cutPoint2 > 0:
                    #     tour = []
                    #     if cutPoint2 == self.cityNum - 1:
                    #         tour.extend(children[j].tour[0: cutPoint1])
                    #         # for k in range(cutPoint1)
                    #         #     tour.append(children[j].tour[k])
                    #     else:
                    #         tour.extend(children[j].tour[0: cutPoint1])
                    #         tour.extend(children[j].tour[cutPoint2 + 1: self.cityNum])
                    #         # for k in range(self.cityNum):
                    #         #     if k < cutPoint1 or k > cutPoint2:
                    #         #         tour.append(children[j].tour[k])
                    #
                    # position = random.randint(len(tour) - 1)
                    # if position == 0:
                    #     for k in range(cutPoint1, cutPoint2 + 1, 1):
                    #         tour.insert(0, children[j].tour[cutPoint1 + cutPoint2 - k])
                    # elif position == len(tour) - 1:
                    #     tour.extend(children[j].tour[cutPoint1: cutPoint2 + 1])
                    #     # for k in range(cutPoint1, cutPoint2 + 1, 1):
                    #     #     tour.append(children[j].tour[k])
                    # else:
                    #     for k in range(cutPoint1, cutPoint2 + 1, 1):
                    #         tour.insert(position, children[j].tour[k])
                    #
                    # children[j] = tour

            self.nextGeneration[i] = children[0]
            self.nextGeneration[i + 1] = children[j]

        for k in range(self.N):
            self.chromosomes[k] == self.nextGeneration[k].clone()

    def printOptimal(self):
        print 'The best fitness is: %.5f' % self.bestFitness
        print 'The best tour length is: %d' % self.bestLength
        print 'The best tour is: '
        print self.bestTour

    def outputResults(self):
        self.printOptimal()


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
            # ga = GA(N=2, cityNum=52, MAX_GEN=100, p_c_t=0.95, p_m_t=0.75, filename='');
            # ga.solve()

            [tour1, tour2] = GA.OX1(tour1=[1, 2, 3, 4, 5, 6, 7, 8], tour2=[2, 4, 6, 8, 7, 5, 3, 1])
            print tour1
            print tour2
            tour = GA.DM(tour=[1, 2, 3, 4, 5, 6, 7, 8])
            print tour
        except getopt.error, msg:
            raise Usage(msg)
            # more code, unchanged
    except Usage, err:
        print >> sys.stderr, err.msg
        print >> sys.stderr, "for help use --help"
        return 2


if __name__ == "__main__":
    sys.exit(main())

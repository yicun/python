#!/usr/bin/python
# -*- coding: utf-8 -*-
#
# 求解一个简单的一维函数f(x) = -(x-1)^2+4, x的取值范围为[-1,3]最大值为例
#

import sys
import getopt
import random


class GeneticAlgorithm(Exception):
    pop = [[]]
    pop_new = []
    fitness_value = []
    fitness_table = []
    fitness_avg = []
    best_fitness = 0.0
    best_generation = -1
    best_individual = []
    curr_generation = 0

    def __init__(self, pop_size=20, chromo_size=16, generation_size=200, cross_rate=0.6, mutate_rate=0.01, elitism=False):
        self.pop_size = pop_size
        self.chromo_size = chromo_size
        self.generation_size = generation_size
        self.cross_rate = cross_rate
        self.mutate_rate = mutate_rate
        self.elitism = elitism

    def initialize(self):
        if self.chromo_size is None or self.chromo_size < 1:
            return None
        self.pop = [[0 for col in range(self.chromo_size)] for row in range(self.pop_size)]
        for row in range(self.pop_size):
            for col in range(self.chromo_size):
                self.pop[row][col] = random.randint(0, 1)

        self.fitness_value = [0.0 for row in range(self.pop_size)]
        self.fitness_table = [0.0 for row in range(self.pop_size)]
        self.best_fitness = 0.0
        self.best_generation = -1
        self.best_individual = [0.0 for col in range(self.chromo_size)]

    def fitness(self):
        for row in range(self.pop_size):
            for col in range(self.chromo_size):
                if self.pop[row][col] == 1:
                    self.fitness_value[row] += 2 ** (col - 1)

            self.fitness_value[row] = -1 + self.fitness_value[row] * (3.0 - (-1.0)) / (2 ** (self.chromo_size - 1))
            self.fitness_value[row] = -(self.fitness_value[row] - 1.0) ** 2 + 4
            if row == 0:
                self.fitness_table[row] = self.fitness_value[row]
            else:
                self.fitness_table[row] = self.fitness_table[row - 1] + self.fitness_value[row]
        self.fitness_avg = self.fitness_table[self.pop_size - 1] / self.pop_size

    def rank(self):
        for row1 in range(self.pop_size):
            for row2 in range(row1 + 1, self.pop_size, 1):
                if self.fitness_value[row1] > self.fitness_value[row2]:
                    tmp = self.fitness_value[row1]
                    self.fitness_value[row1] = self.fitness_value[row2]
                    self.fitness_value[row2] = tmp
                    tmp_row = self.pop[row1]
                    self.pop[row1] = self.pop[row2]
                    self.pop[row2] = tmp_row
        for row in range(self.pop_size):
            if row == 0:
                self.fitness_table[row] = self.fitness_value[row]
            else:
                self.fitness_table[row] = self.fitness_table[row - 1] + self.fitness_value[row]
        self.fitness_avg = self.fitness_table[self.pop_size - 1] / self.pop_size

        if self.fitness_value[self.pop_size - 1] > self.best_fitness:
            self.best_fitness = self.fitness_value[self.pop_size - 1]
            self.best_generation += 1
        self.best_individual = self.pop[self.pop_size - 1]

    def selector(self):
        for row in range(self.pop_size):
            r = random.random() * self.fitness_table[self.pop_size - 1]
            first = 0
            last = self.pop_size - 1
            mid = int((first + last) / 2)
            idx = -1
            while (first <= last) and (idx == -1):
                if r > self.fitness_table[mid]:
                    first = mid
                elif r < self.fitness_table[mid]:
                    last = mid
                else:
                    idx = mid
                    break
                mid = int((last + first) / 2)
                if (last - first) == 1:
                    idx = last
                    break
            self.pop_new.append(self.pop[idx])
        if self.elitism:
            p = self.pop_size - 1
        else:
            p = self.pop_size

        for row in range(p):
            self.pop[row] = self.pop_new[row]

    def crossover(self, cross_rate=None):
        for row in range(0, self.pop_size, 2):
            if random.random() < cross_rate:
                cross_pos = random.randint(0, self.chromo_size - 1)
                if cross_pos == 0 or cross_pos == 1:
                    continue
                for col in range(cross_pos, self.chromo_size, 1):
                    temp = self.pop[row][col]
                    self.pop[row][col] = self.pop[row + 1][col]
                    self.pop[row + 1][col] = temp

    def mutation(self, mutate_rate=None):
        for row in range(0, self.pop_size):
            if random.random() < mutate_rate:
                mutate_pos = random.randint(0, self.chromo_size - 1)
                if mutate_pos == 0:
                    continue
                self.pop[row][mutate_pos] = 1 - self.pop[row][mutate_pos]

    def run(self):
        self.initialize()
        for i in range(self.generation_size):
            self.fitness()
            print 'iter:%d, fitness_avg:%f' % (i, self.fitness_avg)
            self.rank()
            self.selector()
            self.crossover()
            self.mutation()
        print self.best_individual
        print self.best_fitness
        print self.best_generation

    # def plotGA(self, generation_size = 1):
    #     x = range(generation_size)
    #     y = self.fitness_avg;
    #     plot(x,y)

    @staticmethod
    def output(arr=None):
        if arr is not None:
            row_num = len(arr)
            if row_num is not None and row_num > 1:
                for i in range(row_num):
                    row = arr[i]
                    print row


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

            ga = GeneticAlgorithm(pop_size=20, chromo_size=16, generation_size=200, cross_rate=0.8, mutate_rate=0.1, elitism=True)
            ga.run()
            print
        except getopt.error, msg:
            raise Usage(msg)
            # more code, unchanged
    except Usage, err:
        print >> sys.stderr, err.msg
        print >> sys.stderr, "for help use --help"
        return 2


if __name__ == "__main__":
    sys.exit(main())

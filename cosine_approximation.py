import math
import random
import numpy as np
import matplotlib.pyplot as plt
import sys
import copy

class Solution:
    def __init__(self, geneList, graph):
        self.geneList = geneList
        self.graph = graph

    def result_to(self, x):
        result = 0
        i = len(self.geneList) - 1
        for a in self.geneList:
            result += a*x**i
            i -= 1
        return result
    
    def derivate(self, x):
        result = 0
        expoent = len(self.geneList) - 1
        tam = len(self.geneList)
        for i in range(tam - 1):
            result = result + self.geneList[i]*expoent*x**(expoent-1)
            expoent = expoent - 1
        return result
    
    def genScore(self):
        # range to evaluate
        x = np.arange(-np.pi,np.pi,0.1)   # start,stop,step
        resultado2 = 0
        resultado3 = 0

        for u in x:
            resultado2 = resultado2 + abs((-np.sin(u)) - self.derivate(u))
            resultado3 = resultado3 + abs((np.cos(u)) - self.result_to(u))

        resultado = (resultado2 + resultado3)/len(x)


        self.score = 1.0/resultado
    
    def genScorePrint(self):
        # range to evaluate
        x = np.arange(-np.pi,np.pi,0.1)   # start,stop,step

        resultado2 = 0
        resultado3 = 0
        for u in x:
            resultado2 = resultado2 + abs((-np.sin(u)) - self.derivate(u))
            resultado3 = resultado3 + abs((np.cos(u)) - self.result_to(u))

        print("diferença da derivada: {}".format(resultado2))
        print("diferença das posições: {}".format(resultado3))
        resultado = (resultado2 + resultado3)/len(x)
        score = 1.0/resultado
        print("score: {}".format(score))

        self.score = 1.0/resultado

def plotGraphs(population, x):
    for solution in population:
        solution.graph.set_ydata(solution.result_to(x))
    plt.draw()
    plt.pause(1e-3)

def plotFirstGraphs(population, x):
    for solution in population:
        solution.graph.set_ydata(solution.result_to(x))
    plt.plot(x, np.cos(x), label="cos")
    plt.tight_layout()
    plt.ylim(-4, 4)
    plt.draw()
    plt.pause(1e-3)

# single point crossover - bad resultas in this case
def spCo(s0: Solution, s1: Solution, orderSize: int):
        newGene0 = []
        newGene0 = s0.geneList[0:orderSize//2 + 1]
        newGene0 = np.concatenate((newGene0, s1.geneList[orderSize//2+1:orderSize+1]), axis = 0)
        newSolution0 = Solution(newGene0, s0.graph)

        newGene1 = []
        newGene1 = s1.geneList[0:orderSize//2 + 1]
        newGene1 = np.concatenate((newGene1, s0.geneList[orderSize//2+1:orderSize+1]), axis = 0)
        newSolution1 = Solution(newGene1, s1.graph)

        return newSolution0, newSolution1

# uniform crossover
def uCs(s0: Solution, s1: Solution, orderSize: int):
        newGene0 = copy.deepcopy(s0.geneList)
        newGene1 = copy.deepcopy(s1.geneList)
        tam = len(newGene1)
        for i in range(tam):
            if(i%2 == 0):
                newGene1[i] = s0.geneList[i]
                newGene0[i] = s1.geneList[i]
                
        newSolution0 = Solution(newGene0, s0.graph)
        newSolution1 = Solution(newGene1, s1.graph)
        return newSolution0, newSolution1

def genInitialPopulation(size: int, order: int, minRandom :int, maxRandom: int, x):
    population = []
    y = []
    for count in x:
        y.append(0.0)
    for i in range(size):
        geneList = []
        for j in range(order+1):
            geneList.append(random.randint(minRandom,maxRandom) * random.random())
        population.append(Solution(geneList, plt.plot(x, y, label="solution " + str(i) )[0]))
    return population

def genRandomSolution(s: Solution, order: int,  minRandom: int, maxRandom:int ):
    geneList = []
    for i in range(order+1):
        geneList.append(random.randint(minRandom,maxRandom) * random.random())
    return Solution(geneList, s.graph)


def printPopulation(population):
    print("############ SOLUTIONS ##################")
    for solution in population:
        print("[", end="")
        for num in solution.geneList:
            print(str(num)+ " ", end="")
        print("]")
    print("################################################")

def main():
    plt.style.use('fivethirtyeight')
    plt.ion()

    # Set X range to be printed
    x = np.arange(-4*np.pi,4*np.pi,0.1)   # start,stop,step

    # control variables
    minRandom = -1
    maxRandom = 1
    populationSize = 50 # even value
    population = []
    order = 5           # odd value
    targetScore = 10.0
    maxScore = sys.float_info.min
    generation = 0

    population = genInitialPopulation(populationSize, order, minRandom, maxRandom, x=x)

    # checks the first population

    # gen score to all individuals
    for individual in population:
        individual.genScore()

    # order according to score
    population.sort(key=lambda x: x.score, reverse=True)

    maxScore = population[0].score
    plotFirstGraphs(population, x)

    print("running...")
    # main loop
    while(maxScore < targetScore):

        # generates new population
        tmpPopulation = []
        # saves the last populations best solution
        tmpPopulation.append(population[0])

        # crossover
        for i in range(1, len(population)//2, 1 ):
            newSolution0, newSolution1 = uCs(population[i-1], population[i], order)
            tmpPopulation.append(newSolution0)
            tmpPopulation.append(newSolution1)

        # two new fully random solutions
        tmpPopulation.append(genRandomSolution(population[len(population) - 2], order, minRandom, maxRandom))
        tmpPopulation.append(genRandomSolution(population[len(population) - 1], order, minRandom, maxRandom))

        # apply mutation
        for i in range(2,len(population)):
            tmpPopulation[i].geneList[random.randint(0,order)] += random.random() * random.randint(-1, 1)
            tmpPopulation[i].geneList[random.randint(0,order)] += random.random() * random.randint(-1, 1)

        # updates population
        for i in range(len(population)):
            population[i].geneList = tmpPopulation[i].geneList

        for individual in population:
            individual.genScore()

        population.sort(key=lambda x: x.score, reverse=True)
        maxScore = population[0].score
        generation += 1

        # prints plot the graph for every 100 generations
        if(generation%100 == 0):
            print("generation: " + str(generation))
            plotGraphs(population, x)
    
    plt.cla()
    print("fim")
    input()
    print("generation: " + str(generation))
    population[0].genScorePrint()
    
    print(population[0].geneList)

    
    plt.plot(x, population[0].result_to(x))
    plt.plot(x, np.cos(x), label="cos")
    plt.tight_layout()
    plt.ylim(-4, 4)
    plt.draw()
    plt.pause(1e-3)

    input()
        

if __name__ == "__main__":
    main()

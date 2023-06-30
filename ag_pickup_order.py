import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

x = np.array([0, 47.95, 49.6, 54.25, 47.8, 37.375, 39.125, 34.5, 38.25])
y = np.array([10, 19.25, 8.45, 4.65, 8.45, 12.2, 12.2, 15.3, 16.55])

plt.scatter(x, y)
plt.xlabel("x")
plt.ylabel("y")
plt.show()

# number of cities / points
m = len(x)

# number of chromosomes in population
n = 20

# maximum generation
N = 100

# distance matrix
d = np.zeros((m, m), dtype=int)

for i in range(m):
    for j in range(m):
        d[i, j] = np.sqrt((x[i] - x[j])**2 + (y[i] - y[j])**2)

d

def createPopulation():
    pop = np.zeros((n, m), dtype=int)

    for i in range(n):
        pop[i] = np.random.permutation(m)

    pop = pd.DataFrame(pop)

    return pop

def fitness(pop):
    fitness = np.zeros(n, dtype=int)

    for k in range(n):
        a = pop.loc[k]

        b = 0
        for i in range(0, m-1):
            b += d[a[i], a[i+1]]
        b += d[a[m-1], a[0]]

        fitness[k] = b

    pop['fitness'] = fitness

    return pop

# Belum dirubah
def randomSelection():
    position = np.random.permutation(n)

    return position[0], position[1]

# Belum dirubah
def crossover(pop):
    popc = pop.copy()
    for i in range(n):
        a, b = randomSelection()
        x = (pop.loc[a] + pop.loc[b])/2
        popc.loc[i] = x

    return popc

def mutation(pop):
    popm = pop.copy()

    for i in range(n):
        position = np.random.permutation(m)
        a = position[0]
        b = position[1]
        temp = popm.loc[i][a]
        popm.loc[i][a] = popm.loc[i][b]
        popm.loc[i][b] = temp

    return popm

def combinePopulation(pop, popm):
    popAll = pop.copy()
    popAll = popAll.append(popm)

    popAll = popAll.drop_duplicates()

    popAll.index = range(len(popAll))

    return popAll

def sort(popAll):
    popAll = popAll.sort_values(by=['fitness'])

    popAll.index = range(len(popAll))

    return popAll

def elimination(popAll):
    pop = popAll.head(n)

    return pop

def plotSolution(pop):
    solution = pop.loc[0]
    solution = solution.to_numpy()

    a = np.zeros(m+1, dtype=int)
    b = np.zeros(m+1, dtype=int)

    for i in range(m):
        a[i] = x[solution[i]]
        b[i] = y[solution[i]]

    a[m] = a[0]
    b[m] = b[0]

    plt.plot(a, b, marker = 'o')
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()

pop = createPopulation()
pop = fitness(pop)
print('Solusi pada populasi awal')
print(pop.head(1))
plotSolution(pop)

for i in range(1, N+1):
    #popc = crossover(pop)
    #popc = fitness(popc)

    popm = mutation(pop)
    popm = fitness(popm)

    popAll = combinePopulation(pop, popm)

    popAll = sort(popAll)

    pop = elimination(popAll)

    print()
    print('Solusi terbaik pada populasi generasi ke-'+ str(i))
    print(pop.head(1))
    plotSolution(pop)

print()
print('Solusi terbaik pada populasi akhir')
print(pop.head(1))
plotSolution(pop)


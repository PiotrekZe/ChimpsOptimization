import numpy as np
import random


class ChimpOptimization:

    def __init__(self, population, iterations, dimension, low_value, high_value, function):
        self.population = population
        self.iterations = iterations
        self.dimension = dimension
        self.low_value = low_value
        self.high_value = high_value
        self.function = function

    def initialization(self):
        self.chimps = np.random.uniform(self.low_value, self.high_value, (self.population, self.dimension))

    def __divide_groups(self):  # potem trzeba to posortować jeszcze
        n_groups = 4
        temp_size = self.population // n_groups
        num_left = self.population % n_groups
        groups = []

        for i in range(n_groups):
            size = temp_size
            if num_left != 0:
                size += 1
                num_left -= 1
            groups += [i + 1] * size

        self.groups = np.array(groups)

    def __calculate_fintess(self):
        fitness = np.zeros(self.population)

        for i in range(self.population):
            fitness[i] = self.function(self.chimps[i])
        return fitness

    def __create_chimps(self, tab_idx):
        self.Attacker = self.chimps[tab_idx[0]]
        self.Chaser = self.chimps[tab_idx[1]]
        self.Barrier = self.chimps[tab_idx[2]]
        self.Driver = self.chimps[tab_idx[3]]

    def __f(self, t):
        return np.array([2.5 - t ** 2 * (2.5 / self.iterations ** 2)] * self.dimension)

    def __create_varaibles(self, t):
        r1 = np.full(self.dimension, np.random.random(1))
        r2 = np.full(self.dimension, np.random.random(1))
        f = self.__f(t)
        m = np.full(self.dimension, np.random.random(1))  # na razie po prostu randomowa wartość
        c = 2 * r2
        a = 2 * f * r1 - f

        return r1, r2, f, m, c, a

    def solve(self):
        self.initialization()  # mamy małpiszony (populacja, wymiary)
        self.__divide_groups()
        fitness_chimps = self.__calculate_fintess()  # fitness dla kazdego małpiszona
        idx_fitness = np.argsort(fitness_chimps)  # posortowane wartości małpiszonów rosnaco
        self.__create_chimps(idx_fitness)  # cztery pierwsze małpiszony

        for t in range(self.iterations):
            for i in range(self.population):
                # wartości: r, f, m, c, a
                r1, r2, f, m, c, a = self.__create_varaibles(t)

                # print(r1, r2, f, m, c, a)

                d_Attacker = np.abs(c * self.Attacker - m * self.chimps[i])
                d_Barrier = np.abs(c * self.Barrier - m * self.chimps[i])
                d_Chaser = np.abs(c * self.Chaser - m * self.chimps[i])
                d_Driver = np.abs(c * self.Driver - m * self.chimps[i])

                x1 = self.Attacker - a * d_Attacker
                x2 = self.Barrier - a * d_Barrier
                x3 = self.Chaser - a * d_Chaser
                x4 = self.Driver - a * d_Driver

                self.chimps[i] = np.clip((x1 + x2 + x3 + x4) / 4, self.low_value, self.high_value)

            fitness_chimps = self.__calculate_fintess()
            idx_fitness = np.argsort(fitness_chimps)
            self.__create_chimps(idx_fitness)

            print("Position:  ", self.Attacker, "  value:  ", self.function(self.Attacker))
            # print(self.function(self.Attacker))


def function1(tab):  # kwadratowa
    return (tab[0] + 2) ** 2


# chimp_optimization = ChimpOptimization(50, 250, 1, -1000000000, 10000000000000, function1) #np.random.uniform nie lubi np.inf - do sprawdzenia
# chimp_optimization.solve()

def function2(tab):  # Beale function, zbiega do 0.13,0.0055 itd, ale czasem e-05
    return (1.5 - tab[0] + tab[0] * tab[1]) ** 2 + (2.25 - tab[0] + tab[0] * (tab[1]) ** 2) ** 2 + (
                2.625 - tab[0] + tab[0] * (tab[1]) ** 3) ** 2


# chimp_optimization = ChimpOptimization(50, 250, 2, -4.5, 4.5, function2)
# chimp_optimization.solve()

def function3(tab):  # Rastrigin, zbiega do 0.0, wartość zatrzymuje się na e-14 i potem 0.0 od razu
    A = 10
    n = len(tab)
    return A * n + np.sum(np.power(tab, 2) - A * np.cos(2 * np.pi * tab))


# chimp_optimization = ChimpOptimization(50, 100, 10, -5.12, 5.12, function3)
# chimp_optimization.solve()

def function4(tab):  # Goldstein-Price function, zbiega do okolic 3, ale czasem ma około 30 - coś tu musi nie
    # grać bo połowa ma 30 a połowa 3, dosłowanie raz jedno raz drugie -coś moze się psuć dla ujemnych, dla losowych zmiennych
    x = tab[0]
    y = tab[1]
    a = (30 + np.power(2 * x - 3 * y, 2) * (
                18 - 32 * x + 12 * np.power(x, 2) + 48 * y - 36 * x * y + 27 * np.power(y, 2)))
    b = (1 + np.power(x + y + 1, 2) * (19 - 14 * x + 3 * np.power(x, 2) - 14 * y + 6 * x * y + 3 * np.power(y, 2)))
    return a * b


# chimp_optimization = ChimpOptimization(50, 250, 2, -4.5, 4.5, function4)
# chimp_optimization.solve()

def function5(tab):  # Rosenbrock function, nie działa - zbiega do 9 około, a czasami wskoczy do 0. Sprawdzić
    value = 0
    for i in range(len(tab) - 1):
        value += 100 * np.power(tab[i + 1] - np.power(tab[i], 2), 2) + np.power(1 - tab[i], 2)
    return value


# chimp_optimization = ChimpOptimization(50, 250, 50, -100, 100, function5)
# chimp_optimization.solve()

def function6(tab):  # Sphere function, zbiega do e-70 około
    return np.sum(np.power(tab, 2))


# chimp_optimization = ChimpOptimization(50, 250, 10, -100, 100, function6)
# chimp_optimization.solve()

def function7(tab):  # Bukin function N.6
    return 100 * np.sqrt(np.abs(tab[1] - 0.01 * np.power(tab[0], 2))) + 0.01 * np.abs(tab[0] + 10)

# chimp_optimization = ChimpOptimization(50, 250, 2, -15, 4, function7)
# chimp_optimization.solve()
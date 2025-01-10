import numpy as np
import scipy as sp
import random
import math
from geopy import distance

#формулировка задачи

def F(w, y):
    M = len(w)
    sum = 0
    for i in range(M):
        sum += w[i] * y[i]
    return sum

M = 100
n = 5
p = [0.0, 0.25, 0.5, 0.75, 1.0]
W = [random.choice(p) for i in range(M)]
R = 500
x_poi = [round(random.uniform(59.919343, 59.961564), 6) for i in range(M)]
y_poi = [round(random.uniform(30.179292, 30.306886), 6) for i in range(M)]
geo = [0.0 for i in range(M)]
for i in range(M):
    geo[i] = (x_poi[i], y_poi[i])

#вычисление матрицы D - сделать ф-ю
def dist_matrix(geo, r):
    M = len(geo)
    #dist = [[0.0 for j in range(M)] for i in range(M)]
    D = [[0.0 for j in range(M)] for i in range(M)]
    
    for i in range(M):
        for j in range(M):
            D[i][j] = distance.great_circle(geo[i], geo[j]).km * 1000

    for i in range(M):
        for j in range(M):
            if D[i][j] <= r:
                D[i][j] = 1
            else:
                D[i][j] = 0
    return D

D = dist_matrix(geo, R)

#инициализация X и заполнение X0 - сделать ф-ю
def init(M, n):
    X = [0 for i in range(M)]
    z = [i for i in range(M)]
    if n <= M:
        while n > 0:
            ch = random.choice(z)
            z.remove(ch)
            X[ch] = 1
            n -=1
    return X

#расчёт Y - сделать ф-ю
def coverage(d, x):
    M = len(x)
    Y = [0 for i in range(M)]
    for i in range(M):
        for j in range(M):
            Y[j] += d[i][j] * x[i]
    return Y

#Метод отжига

#расчёт температуры на k-ом шаге
def temperature(t, k):
    T = t / k
    return T

#генерация близкого решения к решению на прошлом шаге
def neighbour(u):
    p1 = random.random()
    p2 = random.random()
    z = random.randrange(len(u)+1)
    if z >= len(u)-1:
        Z = random.randrange(1, len(u[:z]))
        N = np.concatenate([np.roll(u[:z], Z), u[z:]]) #движение левой части
    elif z <= 1:
        Z = random.randrange(1, len(u[z:]))
        N = np.concatenate([u[:z], np.roll(u[z:], Z)]) #движение правой части
    else:
        if p1 < p2:
            Z = random.randrange(1, len(u[:z]))
            N = np.concatenate([np.roll(u[:z], Z), u[z:]]) #движение левой части
        else:
            Z = random.randrange(1, len(u[z:]))
            N = np.concatenate([u[:z], np.roll(u[z:], Z)]) #движение правой части
    return N

#вероятность перехода
def probability(f_old, f_new, t_k):
    P = math.exp((f_new - f_old) / t_k)
    return P

#вводные параметры и сам алгоритм
k_max = 100
T0  = 1000

def simulated_annealing(n, W, D, T0, k_max):
    k = 0
    M = len(W)
    X = init(M, n)
    T = temperature(T0, k+1)
    while k < k_max:
        if T > 0 and k_max > 0:
            N = neighbour(X)
            F_old = F(W, coverage(D, X))
            F_new = F(W, coverage(D, N))
            if F_new > F_old:
                X = N
                k += 1
                T = temperature(T, k+1)
            else:
                if random.random() < probability(F_old, F_new, T):
                    X = N
                    k += 1
                    T = temperature(T, k+1)
                else:
                    k += 1
                    T = temperature(T, k+1)
    return X, F(W, coverage(D, X))

#Генетический алгоритм

#генерация популяции
def population(M, n, L):
    return np.array([init(M, n) for i in range(L)])

#выбор родителей
def parents(x, D, W):
    #x = population(M, n, L)
    l = len(x)
    f = [0 for i in range(l)]
    c = [0 for i in range(l)]
    sum = 0
    for i in range(l):
        f[i] = F(W, coverage(D, x[i]))
    for i in range(l):
        c[i] = f[i] - min(f)
        sum += c[i]

    pos = len([i for i in c if i > 0.0])
    k = [0 for i in range(pos)]

    for i in range(pos):
        cd = np.array([c[i]/sum for i in range(l)])
        cs = np.array(np.cumsum(cd))
        rnd = random.random()
        cs = np.append(cs, rnd)
        cs = np.sort(cs)
        k[i] = np.where(cs == rnd)[0][0]
        sum = sum - c[k[i]]
        c[k[i]] = 0
        k[i] = x[k[i]]
    return np.asarray(k)

#кроссинговер
def crossover(X):
    x = X.tolist()
    l = len(x)
    offs = np.array([[0 for i in range(M)] for j in range(2*l)])
    for i in range(int(l-1)):
        z = random.randrange(1, M)
        offs[2*i] = x[i][:z] + x[i+1][z:]
        offs[2*i+1] = x[i+1][:z] + x[i][z:]
    offs[2*l-2] = x[l-1][:z] + x[0][z:]
    offs[2*l-1] = x[0][:z] + x[l-1][z:]
    
    sum_offs = np.array([0 for i in range(2*l)])
    for i in range(2*l):
        sum_offs[i] = np.sum(offs[i])
        while sum_offs[i] != n:
            if sum_offs[i] > n:
                k = np.where(offs[i] == 1)[0].tolist()
                m = random.choice(k)
                offs[i][m] = 0
                sum_offs[i] = np.sum(offs[i])
            elif sum_offs[i] < n:
                k = np.where(offs[i] == 0)[0].tolist()
                m = random.choice(k)
                offs[i][m] = 1
                sum_offs[i] = np.sum(offs[i])
    return offs

#мутации
def mutation(x, P):
    l = len(x)
    for i in range(l):
        if P > random.random():
            k1 = np.where(x[i] == 1)[0].tolist()
            k2 = np.where(x[i] != 1)[0].tolist()
            m1 = random.choice(k1)
            m2 = random.choice(k2)
            x[i][m1], x[i][m2] = x[i][m2], x[i][m1]
    return x

#отбор потомков в новое поколение
def selection(x):
    #x = mutation(M, n, D, W, L, P)
    l = len(x)
    next_gen = np.array([[0 for i in range(M)] for j in range(int(l/2))])
    f = [0 for i in range(l)]
    for i in range(l):
        f[i] = F(W, coverage(D, x[i]))
    for i in range(int(l/2)):
        if f[2*i] > f[2*i+1]:
            next_gen[i] = x[2*i]
        else:
            next_gen[i] = x[2*i+1]
    return next_gen

#вводные параметры и сам алгоритм
L = 100
P = 0.9
k = 3

def genetic_algorithm(M, n, D, W, L, P, k):
    x = population(M, n, L)
    while len(x) > k:
        x = selection(mutation(crossover(parents(x, D, W)), P))
    l = len(x)
    f = [0 for i in range(l)]
    for i in range(l):
        f[i] = F(W, coverage(D, x[i]))
        max_f = max(f)
    u = np.where(f == max_f)[0].tolist()
    X = np.array([x[u[i]] for i in range(len(u))])
    return X, max_f

temps = [10**(i) for i in range(5)]
k_maxs = [(i+1)*10 for i in range(10)]

pops = [(i+1)*20 for i in range(10)]
stops = [(i+1)*2 for i in range(10)]
#обычное решение ЗЛП
#import pulp

#model = pulp.LpProblem("EVCSLocation", pulp.LpMaximize)
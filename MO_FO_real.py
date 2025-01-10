import numpy as np
import random
import math

from geopy import distance

import pandas as pd

import matplotlib.pyplot as plt
import pylab
import folium
import time

from spopt.locate import MCLP
from spopt.locate.util import simulated_geo_points
import geopandas as gpd
import pulp
import spaghetti

#формулировка задачи
def F(w, y):
    M = len(w)
    sum = 0
    for i in range(M):
        sum += w[i] * y[i]
    return sum

#вычисление матрицы D
def dist_matrix(geo, r):
    M = len(geo)
    D = [[0.0 for j in range(M)] for i in range(M)]
    
    for i in range(M):
        for j in range(M):
            D[i][j] = distance.great_circle(geo[i], geo[j]).km * 1000
            if D[i][j] <= r:
                D[i][j] = 1
            else:
                D[i][j] = 0
    return D

#инициализация X и заполнение X0
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

#расчёт Y
def coverage(d, x):
    M = len(x)
    Y = [0 for i in range(M)]
    for i in range(M):
        for j in range(M):
            Y[j] += d[i][j] * x[i]
            if Y[j] > 0:
                Y[j] = 1
            else: 
                Y[j] = 0
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

def simulated_annealing(T0, k_max):
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
            else:
                if random.random() < probability(F_old, F_new, T):
                    X = N
                    k += 1
                else:
                    k += 1
        T = temperature(T, k+1)
    return X, F(W, coverage(D, X))

#Генетический алгоритм

#генерация популяции
def population(L):
    return np.array([init(M, n) for i in range(L)])

#выбор родителей
def parents(x):
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
def mutation(x, p):
    l = len(x)
    for i in range(l):
        if p > random.random():
            k1 = np.where(x[i] == 1)[0].tolist()
            k2 = np.where(x[i] != 1)[0].tolist()
            m1 = random.choice(k1)
            m2 = random.choice(k2)
            x[i][m1], x[i][m2] = x[i][m2], x[i][m1]
    return x

#отбор потомков в новое поколение
def selection(x):
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

def genetic_algorithm(L, p, k):
    x = population(L)
    while len(x) > k:
        x = selection(mutation(crossover(parents(x)), p))
    l = len(x)
    f = [0 for i in range(l)]
    for i in range(l):
        f[i] = F(W, coverage(D, x[i]))
        max_f = max(f)
    u = np.where(f == max_f)[0].tolist()
    X = np.array([x[u[i]] for i in range(len(u))])
    return X, max_f

#подготовка выгруженных данных
def data_preparation(name):
    data = pd.read_csv(f"{name}")
    data = data[['osm_id', 'name', 'fuel', 'building', 'public_transport', 'amenity', 'leisure', 'parking', 'longitude', 'latitude']]
    data = data.fillna('-')
    data['description'] = data['fuel'] + data['building'] + data['public_transport'] + data['amenity'] + data['leisure'] + data['parking']
    return data

data = data_preparation("MO_FO.csv")

#создание словаря
weight = [0.0 for i in range(data.shape[0])]
data.insert(11, "weight", weight)

weight_dict = {
    1.0: {
        'fuel': ['gas'], 
        'building': ['house', 'residential', 'apartments', 'hotel', 'dormitory', 'parking', 'fuel'], 
        'public_transport': [], 
        'amenity': ['parking', 'fuel', 'dormitory'], 
        'leisure': ['resort'], 
        'parking': ['lane', 'street_side', 'surface', 'underground', 'multi-storey', 'rooftop', 'sheds', 'carports']
    }, 
    0.75: {
        'fuel': [], 
        'building': ['service', 'public', 'commercial', 'office', 'school', 'kindergarten', 'policlinic', 'university', 'hospital', 'civic', 'government', 'college', 'polyclinic', 'offices', 'doctors', 'museum', 'clinic', 'central_office', 'community_centre'], 
        'public_transport': [], 
        'amenity': ['school', 'doctors', 'dentist', 'clinic', 'music_school', 'bank', 'public_building', 'social_facility', 'college', 'university', 'veterinary', 'hospital', 'library', 'courthouse', 'community_centre', 'kindergarten', 'language_school', 'townhall', 'business_center', 'public', 'nursing_home', 'research_institute', 'childcare'], 
        'leisure': [], 
        'parking': []
    }, 
    0.5: {
        'fuel': [], 
        'building': ['retail', 'store', 'sports_centre', 'shop', 'palace', 'theatre', 'stadium', 'bakehouse', 'bathhouse', 'sports_hall', 'mall', 'workshop', 'sport_centre', 'supermarket', 'market', 'swimming_pool', 'swiming_pool', 'castle'], 
        'public_transport': [], 
        'amenity': ['fast_food', 'cafe', 'theatre', 'arts_centre', 'restaurant', 'driving_school', 'training', 'planetarium', 'cinema', 'marketplace', 'nightclub', 'post_office', 'bar', 'pharmacy', 'events_venue', 'pub', 'hookah_lounge', 'exhibition_centre', 'food_court', 'public_bath', 'workshop'], 
        'leisure': ['park', 'fitness_station', 'sports_centre', 'ice_rink', 'sauna', 'fitness_centre', 'swimming_pool', 'stadium', 'chess_club', 'sports_hall'], 
        'parking': []
    }, 
    0.25: {
        'fuel': [], 
        'building': ['church', 'chapel', 'train_station', 'mosque', 'religious', 'temple', 'synagogue', 'basilica', 'cathedral', 'campanile'], 
        'public_transport': ['station', 'platform'], 
        'amenity': ['bus_station', 'place_of_worship', 'ferry_terminal'], 
        'leisure': [], 
        'parking': []
    }, 
    0.0: {
        'fuel': ['nan'], 
        'building': ['nan', 'yes', 'kiosk', 'bunker', 'static_caravan', 'collapsed', 'entrance', 'garages', 'factory', 'pavilion', 'ruins', 'warehouse', 'detached', 'roof', 'garage', 'stable', 'shelter', 'no', 'gate', 'cabin', 'container', 'industrial', 'construction', 'garbage_shed', 'hangar', 'transformer_tower', 'storage_tank', 'terrace', 'barracks', 'guardhouse', 'manufacture', 'stands', 'transportation', 'houseboat', 'greenhouse', 'shed', 'hut', 'water_tower', 'farm_auxiliary', 'reservoir', 'allotment_house', 'marquee', 'semidetached_house', 'power_substation', 'bridge', 'bungalow', 'depot', 'grandstand', 'toilets', 'ramp', 'abandoned', 'tent', 'gazebo', 'proposed', 'constructionУсть-Славянское', 'шоссе', 'fire_station', 'wall', 'barn', 'boathouse', 'skywalk', 'parking_entrance', 'telephone_exchange', 'cowshed', 'carport', 'power', 'inflated', 'fence', 'lighthouse', 'ship', 'electricity', 'gatehouse', 'part', 'works', 'boiler_house', 'tower', 'air-supported', 'grotto', 'stilt_house', 'research', 'security_booth', 'security_post', 'was:collapsed', 'military', 'demolition', 'outbuilding', 'farm', 'disused', 'boat', 'palace;yes', 'utility', 'retailq', 'fort'], 
        'public_transport': ['nan'], 
        'amenity': ['nan', 'recycling', 'waste_disposal', 'vending_machine', 'police', 'drinking_water', 'shelter', 'bench', 'toilets', 'car_wash', 'security_booth', 'guard', 'fire_station', 'trolley_bay', 'parking_entrance', 'mortuary', 'shower', 'boat_rental', 'ski_rental', 'theatre_stage', 'vehicle_inspection', 'checkpoint', 'bicycle_rental', 'reception_desk', 'rescue_station', 'crematorium', 'stage', 'vacuum_cleaner', 'customs', 'stroller_parking', 'smoking_area', 'fountain'], 
        'leisure': ['nan', 'ice_cream', 'horse_riding', 'pitch', 'trampoline_park', 'bleachers', 'bandstand', 'outdoor_seating', 'playground', 'garden', 'common'], 
        'parking': ['nan', 'underground=roof']
    }
}

#логика проверки и замены весов и добавление весов
weight_items = weight_dict.items()
break_point = False
key = 0.0

for i in range(data.shape[0]):
    break_point = False
    key = 0.0
    for k in range(len(weight_items)):
        for j in range(6):
            if data.iat[i, j+2] in list(list(weight_items)[k][1].items())[j][1]:
                key = list(weight_items)[k][0]
                break_point = True
                break
        if key > data.iat[i, 11]:
            data.iat[i, 11] = key
        if break_point:
            break

data = data[data['weight'] != 0.0]
dlen = data.shape[0]

#входные данные
M = dlen
n = 10
R = 500
W = [data.iat[i, 11] for i in range(M)]
geo = [(data.iat[i, 9], data.iat[i, 8]) for i in range(M)]
D = dist_matrix(geo, R)

#параметры SA
k_max = 100
T0  = 1000

#параметры GA
L = 100
P = 0.9
k = 3

#sa = simulated_annealing(T0, k_max)
#ga = genetic_algorithm(L, P, k)

m = folium.Map([59.972121, 30.368988], zoom_start=13)

#визуализация на карте
def map_visualization(cs_col):
    m = folium.Map([59.972121, 30.368988], zoom_start=13)
    data['CS'] = cs_col
    for i in range(M):
        if data.iat[i, 12] == 1:
            folium.Marker(
                location = [data.iat[i, 9], data.iat[i, 8]],
                icon = folium.Icon(color = "darkblue", icon = "arrow-up", angle = 180),
                tooltip = f"{round(data.iat[i, 9], 6)}, {round(data.iat[i, 8], 6)}"
            ).add_to(m)
            folium.Circle(
                color = "purple",
                location = [data.iat[i, 9], data.iat[i, 8]],
                radius = R,
                fill = True
            ).add_to(m)
    return m

#Что вообще можно изменять? (и наблюдать за временем работы). Каждый вариант алгоритма крутим по 10 раз
# В модели:
    # 1. Количество POI (СПб, ВО, МО ФО) -- Меняем POI и смотрим результаты SA/GA при фиксированных CS и радиусе. Таблица: строки -- количество POI (СПб, ВО, МО ФО), столбцы -- SA и GA с подстолбцами время работы и значение целевой функции. Таких таблиц, к примеру, три штуки: на 10, n-россетей и n-МСК.
POI_dict = ["VO.csv", "MO_FO.csv"]
elapsed_time = []
target_value = []

cs_number = []
r_value = []

    # 2. Количество CS (от единиц до сотен) -- Меняем CS и смотрим то же самое при фиксированных POI и радиусе. Таблица: строки -- количество CS (5, 10, 50, 100, 300), столбцы -- SA и GA. Таких таблиц тоже три: СПб, ВО, МО ФО.
cs_number = [10, 50, 100]
elapsed_time = []
target_value = []

POI_dict = []
r_value = []

    # 3. Радиус (от десятков метров до километра) -- меняем радиус и смотрим то же самое при фиксированных POI и CS. Таблица: строки -- радиус покрытия (10, 25, 125, 500, 1000), столбцы -- SA и GA. Таких таблиц, к примеру, три штуки: на 10, n-россетей и n-МСК.
r_value = [100, 500, 1000]
elapsed_time = []
target_value = []

POI_dict = []
cs_number = []

# В SA:
    # 1. Начальную температуру (от ) -- Меняем T0 и смотрим результаты SA (время и значение целевой функции). 
#От T0 не зависит время работы алгоритма (т.к. это вводная большая температура, чтобы заработал алгоритм, а само время работы контролирует k_max), а значение функции всё-таки зависит:
#То есть по сути своей SA разрешает компромисс между исследованием и эксплуатацией -- при огромной температуре алгоритм распыляется на исследования, но остывая, переходит ко всё более последовательной эксплуатации. Выбрав очень большую температуру, можно регулировать процесс работы алгоритма значением k_max: при относительно (T0) малом k_max алгоритм только начнёт исследовать, как почти тут же завершит работу, почти не начав двигаться к эксплуатации; при большом k_max алгоритм постепенно будет переходить от исследования к эксплуатации, но здесь уже важно не дойти до того случая, когда большую часть времени работы алгоритм будет находиться в состоянии простоя (т.е. выбирать прежнее решение, и сгенерированное новое почти всегда будет хуже прежнего, а если и лучше, то разница значений решения не будет оправдывать время работы алгоритма). При очень малой температуре алгоритм сразу перейдёт к эксплуатации, независимо от k_max; увеличением k_max можно получить только больше шансов на генерацию лучшего решения и больше шансов разово совершать исследование (зависит от random.random()); при очень большом k_max компилятор просто останавливает выполнение программы, т.к. в знаменателе степени экспоненты при вычислении вероятности появляется очень маленькое число, и значения дроби каждого шага стремятся к бесконечности.
    # 2. Максимальное количество шагов алгоритма (от единиц до сотен) -- Меняем k_max и смотрим результаты SA. Таблица: строки -- кол-во шагов, столбцы -- SA при разных фикс. POI, CS, R. Подстолбцы -- время работы и значение функции.
    #Как я понимаю, в каждой конкретной задаче (при разных размерах) необходимо подгонять свои T0 и k_max для достижения больших значений целевой функции. Т.е. берём, к примеру, SA на ВО с фикс. CS и R, ставим небольшое количество шагов, ставим достаточно большую температуру; начинаем подгонять k_max, чтобы алгоритм беспрепятственно переходил от исследования к эксплуатации, достигая наибольшего значения функции и завершал работу за разумное время. 
    # ТАБЛИЦА И ГРАФИК: график -- фикс. большая температура и 1) время к k_max; 2) значение функции к k_max. Такое проделать со всеми вариантами, которые будут в модели
w_bound = sum(W)

T0 = 10000000000000
num = 13
k_maxs = [10*(i+1) for i in range(num)]
elapsed_time = [0 for i in range(num)]
target_value = [0 for i in range(num)]
elapsed_mean = 0
target_mean = 0

for i in range(num):
    for j in range(10):
        s_t = time.perf_counter()
        sa = simulated_annealing(T0, k_maxs[i])
        e_t = time.perf_counter() - s_t
        elapsed_mean += e_t
        target_mean += sa[1]
    elapsed_time[i] = elapsed_mean / 10
    target_value[i] = target_mean / 10 / w_bound
    print('Coverage: ', target_value[i], 'Elapsed time: ', elapsed_time[i], 'T0: ', T0, 'k_max: ', k_maxs[i])
    elapsed_mean = 0
    target_mean = 0

#графики
fig = plt.figure(figsize = (14.0, 5.0))
plt.suptitle('MO Финляндский округ, CS = 10, R = 500, T0 = 10000000000000', weight = 'bold')
pylab.subplot(1, 2, 1)
plt.plot(k_maxs, elapsed_time, 'r')
plt.xlabel('Максимальное кол-во итераций')
plt.xlim(0, 130)
plt.ylabel('Среднее время работы алгоритма')
plt.grid(True)
pylab.subplot(1, 2, 2)
plt.plot(k_maxs, target_value, 'r')
plt.xlabel('Максимальное кол-во итераций')
plt.xlim(0, 130)
plt.ylabel('Доля покрытия')
plt.ylim(0.0, 1.0)
plt.grid(True)
plt.show()

# В GA:
    # 1. Размер популяции (менять от десятков до сотен)
    # 2. Вероятность мутации (просто менять вероятность с шагом, к примеру, 0.1 или 0.05)
    # 3. Граница вырождения (менять от единиц до десятков)

L = 13
P = 0.9
k = 3

num = 10
l = [(4 + i) for i in range(num)]
p = [(0.1 + i*0.1) for i in range(num)]
k_s = [12-i for i in range(num)]
elapsed_time = [0 for i in range(num)]
target_value = [0 for i in range(num)]
elapsed_mean = 0
target_mean = 0

for i in range(num):
    for j in range(10):
        s_t = time.perf_counter()
        ga = genetic_algorithm(l[i], P, k)
        e_t = time.perf_counter() - s_t
        elapsed_mean += e_t
        target_mean += ga[1]
    elapsed_time[i] = elapsed_mean / 10
    target_value[i] = target_mean / 10 / w_bound
    print('Coverage: ', target_value[i], 'Elapsed time: ', elapsed_time[i], 'P: ', P, 'k: ', k, 'l: ', l[i])
    elapsed_mean = 0
    target_mean = 0

#графики
fig = plt.figure(figsize = (14.0, 5.0))
plt.suptitle('MO Финляндский округ, CS = 10, R = 500, K = 3, P = 0.9', weight = 'bold')
pylab.subplot(1, 2, 1)
plt.plot(l, elapsed_time, 'r')
plt.xlabel('Размер популяции')
plt.xlim(4, 13)
plt.ylabel('Среднее время работы алгоритма')
plt.grid(True)
pylab.subplot(1, 2, 2)
plt.plot(l, target_value, 'r')
plt.xlabel('Размер популяции')
plt.xlim(4, 13)
plt.ylabel('Доля покрытия')
plt.ylim(0.0, 1.0)
plt.grid(True)
plt.show()

# Затем по этим таблицам составить графики в matplotlib

#таблицы
table_POI = [[0 for i in range(4)] for j in range(len(POI_dict))]

data = data_preparation("MO_FO.csv")
weight = [0.0 for i in range(data.shape[0])]
data.insert(11, "weight", weight)
weight_items = weight_dict.items()
break_point = False
key = 0.0
for i in range(data.shape[0]):
    break_point = False
    key = 0.0
    for k in range(len(weight_items)):
        for j in range(6):
            if data.iat[i, j+2] in list(list(weight_items)[k][1].items())[j][1]:
                key = list(weight_items)[k][0]
                break_point = True
                break
        if key > data.iat[i, 11]:
            data.iat[i, 11] = key
        if break_point:
            break
data = data[data['weight'] != 0.0]
dlen = data.shape[0]
M = dlen
n = 10
R = 500
W = [data.iat[i, 11] for i in range(M)]
geo = [(data.iat[i, 9], data.iat[i, 8]) for i in range(M)]
D = dist_matrix(geo, R)

T0 = 10000000000000
k_max = 100
L = 12
P = 0.9
k = 3

elapsed_time_sa = [0 for i in range(3)]
target_value_sa = [0 for i in range(3)]
cov_sa = [0 for i in range(3)]
elapsed_mean_sa = 0
target_mean_sa = 0

elapsed_time_ga = [0 for i in range(3)]
target_value_ga = [0 for i in range(3)]
cov_ga = [0 for i in range(3)]
elapsed_mean_ga = 0
target_mean_ga = 0

for e in range(len(r_value)):
    D = dist_matrix(geo, r_value[e])
    for i in range(len(cs_number)):
        n = cs_number[i] #поменять местами n = cs и D = dist (в тетрадке написано)
        for j in range(10):
            s_t = time.perf_counter()
            sa = simulated_annealing(T0, k_max)
            e_t = time.perf_counter() - s_t
            elapsed_mean_sa += e_t
            target_mean_sa += sa[1]
            s_t = time.perf_counter()
            ga = genetic_algorithm(L, P, k)
            e_t = time.perf_counter() - s_t
            elapsed_mean_ga += e_t
            target_mean_ga += ga[1]
        elapsed_time_sa[i] = elapsed_mean_sa / 10
        target_value_sa[i] = target_mean_sa / 10
        cov_sa[i] = target_mean_sa / 10 / w_bound
        elapsed_time_ga[i] = elapsed_mean_ga / 10
        target_value_ga[i] = target_mean_ga / 10
        cov_ga[i] = target_mean_ga / 10 / w_bound
        #print('Elapsed time: ', elapsed_time[i], 'Value: ', target_value[i], 'Coverage: ', cov_sa[i])
        elapsed_mean_ga = 0
        target_mean_ga = 0
        elapsed_mean_sa = 0
        target_mean_sa = 0
        table_results[(3*e)+i][0] = elapsed_time_sa[i]
        table_results[(3*e)+i][1] = target_value_sa[i]
        table_results[(3*e)+i][2] = cov_sa[i]
        table_results[(3*e)+i][3] = elapsed_time_ga[i]
        table_results[(3*e)+i][4] = target_value_ga[i]
        table_results[(3*e)+i][5] = cov_ga[i]

for e in range(len(cs_number)):
    n = cs_number[e]
    for i in range(num):
        D = dist_matrix(geo, r_value[i])
        for j in range(10):
            
        elapsed_time[i] = elapsed_mean / 10
        target_value[i] = target_mean / 10
        cov_ga[i] = target_mean / 10 / w_bound
        print('Value: ', target_value[i], 'Elapsed time: ', elapsed_time[i], 'P: ', P, 'k: ', k, 'l: ', L)
        elapsed_mean = 0
        target_mean = 0
        table_POI[0][2] = elapsed_time[0]
        table_POI[0][3] = target_value[0]

table_CS = [[0 for i in range(4)] for j in range(len(cs_number))]

table_R = [[0 for i in range(4)] for j in range(len(r_value))]

table_results = [[0 for i in range(len(value_dict))] for j in range(len(cs_number)*len(r_value))]

value_dict = ['Т сред. SA', 'F сред. SA', 'Cov. SA', 'T сред. GA', 'F сред. GA', 'Cov. GA']

table(ax, cellText=table_POI, cellLoc='center', rowLabels=POI_dict, rowLoc='left', colLabels=value_dict, colLoc='center', edges='closed', loc='bottom')

table(ax,
          cellText=table_CS, cellColours=None,
          cellLoc='center', colWidths=None,
          rowLabels=cs_number, rowColours=None, rowLoc='left',
          colLabels=value_dict, colColours=None, colLoc='center',
          edges='closed',
          edgeColour=None,
          cellEdgeColours=None,
          rowEdgeColours=None,
          colEdgeColours=None,
          loc='bottom', bbox=None,
          **kwargs)

table(ax,
          cellText=table_R, cellColours=None,
          cellLoc='center', colWidths=None,
          rowLabels=r_value, rowColours=None, rowLoc='left',
          colLabels=value_dict, colColours=None, colLoc='center',
          edges='closed',
          edgeColour=None,
          cellEdgeColours=None,
          rowEdgeColours=None,
          colEdgeColours=None,
          loc='bottom', bbox=None,
          **kwargs)
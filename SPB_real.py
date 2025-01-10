import numpy as np
import scipy as sp
import random
import math

from geopy import distance

import sklearn
import pandas as pd
from sklearn import metrics

import folium
import time

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
data = pd.read_csv("SPB.csv")
data = data[['osm_id', 'name', 'fuel', 'building', 'public_transport', 'amenity', 'leisure', 'parking', 'longitude', 'latitude']]
data = data.fillna('-')
data['description'] = data['fuel'] + data['building'] + data['public_transport'] + data['amenity'] + data['leisure'] + data['parking']

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
n = 300
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

m = folium.Map([59.947931, 30.192691], zoom_start=11)

#визуализация на карте
def map_visualization(cs_col):
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
import numpy as np
import scipy as sp
import random
import math
from geopy import distance

import sklearn
import pandas as pd
from sklearn import metrics

import folium

data = pd.read_csv("SPB.csv")
#print(data)

data = data[['osm_id', 'name', 'fuel', 'building', 'public_transport', 'amenity', 'leisure', 'parking', 'longitude', 'latitude']]
#print(data)

data = data.fillna('-')
#print(data)

data['description'] = data['fuel'] + data['building'] + data['public_transport'] + data['amenity'] + data['leisure'] + data['parking']
#print(data)

#print(data['description'].unique())

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

weight_items = weight_dict.items()

#придумать логику проверки и замены весов
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
W = [data.iat[i, 11] for i in range(data.shape[0])]
#key = next((key for key, value in weight_items if data.iat[i, j+2] in list(list(weight_items)[k][1].items())[j][1]), 0.0)

# weight_dict = {
#     1.0: ['gas', 'house', 'residential', 'apartments', 'hotel', 'dormitory', 'parking', 'fuel', 'parking', 'fuel', 'dormitory', 'resort', 'lane', 'street_side', 'surface', 'underground', 'multi-storey', 'rooftop', 'sheds', 'carports'], 
#     0.75: ['service', 'public', 'commercial', 'office', 'school', 'kindergarten', 'policlinic', 'university', 'hospital', 'civic', 'government', 'college', 'polyclinic', 'offices', 'doctors', 'museum', 'clinic', 'central_office', 'community_centre', 'school', 'doctors', 'dentist', 'clinic', 'music_school', 'bank', 'public_building', 'social_facility', 'college', 'university', 'veterinary', 'hospital', 'library', 'courthouse', 'community_centre', 'kindergarten', 'language_school', 'townhall', 'business_center', 'public', 'nursing_home', 'research_institute', 'childcare'],
#     0.5: ['retail', 'store', 'sports_centre', 'shop', 'palace', 'theatre', 'stadium', 'bakehouse', 'bathhouse', 'sports_hall', 'mall', 'workshop', 'sport_centre', 'supermarket', 'market', 'swimming_pool', 'swiming_pool', 'castle', 'fast_food', 'cafe', 'theatre', 'arts_centre', 'restaurant', 'driving_school', 'training', 'planetarium', 'cinema', 'marketplace', 'nightclub', 'post_office', 'bar', 'pharmacy', 'events_venue', 'pub', 'hookah_lounge', 'exhibition_centre', 'food_court', 'public_bath', 'workshop', 'park', 'fitness_station', 'sports_centre', 'ice_rink', 'sauna', 'fitness_centre', 'swimming_pool', 'stadium', 'chess_club', 'sports_hall'],
#     0.25: ['church', 'chapel', 'train_station', 'mosque', 'religious', 'temple', 'synagogue', 'basilica', 'cathedral', 'campanile', 'station', 'platform', 'bus_station', 'place_of_worship', 'ferry_terminal'],  
#     0.0: ['nan', 'nan', 'yes', 'kiosk', 'bunker', 'static_caravan', 'collapsed', 'entrance', 'garages', 'factory', 'pavilion', 'ruins', 'warehouse', 'detached', 'roof', 'garage', 'stable', 'shelter', 'no', 'gate', 'cabin', 'container', 'industrial', 'construction', 'garbage_shed', 'hangar', 'transformer_tower', 'storage_tank', 'terrace', 'barracks', 'guardhouse', 'manufacture', 'stands', 'transportation', 'houseboat', 'greenhouse', 'shed', 'hut', 'water_tower', 'farm_auxiliary', 'reservoir', 'allotment_house', 'marquee', 'semidetached_house', 'power_substation', 'bridge', 'bungalow', 'depot', 'grandstand', 'toilets', 'ramp', 'abandoned', 'tent', 'gazebo', 'proposed', 'constructionУсть-Славянское шоссе', 'fire_station', 'wall', 'barn', 'boathouse', 'skywalk', 'parking_entrance', 'telephone_exchange', 'cowshed', 'carport', 'power', 'inflated', 'fence', 'lighthouse', 'ship', 'electricity', 'gatehouse', 'part', 'works', 'boiler_house', 'tower', 'air-supported', 'grotto', 'stilt_house', 'research', 'security_booth', 'security_post', 'was:collapsed', 'military', 'demolition', 'outbuilding', 'farm', 'disused', 'boat', 'palace;yes', 'utility', 'retailq', 'fort', 'nan', 'nan', 'recycling', 'waste_disposal', 'vending_machine', 'police', 'drinking_water', 'shelter', 'bench', 'toilets', 'car_wash', 'security_booth', 'guard', 'fire_station', 'trolley_bay', 'parking_entrance', 'mortuary', 'shower', 'boat_rental', 'ski_rental', 'theatre_stage', 'vehicle_inspection', 'checkpoint', 'bicycle_rental', 'reception_desk', 'rescue_station', 'crematorium', 'stage', 'vacuum_cleaner', 'customs', 'stroller_parking', 'smoking_area', 'fountain', 'nan', 'ice_cream', 'horse_riding', 'pitch', 'trampoline_park', 'bleachers', 'bandstand', 'outdoor_seating', 'playground', 'garden', 'common', 'nan', 'underground=roof']
# }

# key = next((key for key, value in weight_dict.items() if  in value), None)

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
k_max = 100
T0  = 1000

def temperature(t, k):
    T = t / k
    return T

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

def probability(f_old, f_new, t_k):
    P = math.exp((f_new - f_old) / t_k)
    return P

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
    return X
import sklearn
import pandas as pd
from sklearn import metrics

data = pd.read_csv("VO.csv")
print(data)

data = data[['osm_id', 'building', 'public_transport', 'amenity', 'leisure', 'parking', 'longitude', 'latitude']]
print(data)

data = data.fillna('-')
print(data)

data['description'] = data['building'] + data['public_transport'] + data['amenity'] + data['leisure'] + data['parking']
print(data)

print(data['description'].unique())
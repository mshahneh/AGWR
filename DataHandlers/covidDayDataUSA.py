import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
import pandas as pd
import numpy as np
import pickle
import os
from storeData import store

path = "../../Data/covidDayDataUSA"
population_data = pd.read_csv("../../Data/covidDayDataUSA/acs2015_county_data.csv")
covid_data = pd.read_csv("../../Data/covidDayDataUSA/07-14-2020.csv")
print(population_data.shape, covid_data.shape)


population_categories = ['County', 'State', 'TotalPop', 'Transit', 'WorkAtHome']
population_data = population_data[population_categories]

covid_data = covid_data[covid_data["Country_Region"] == "US"]
covid_data = covid_data.loc[~covid_data['Province_State'].isin(['Alaska', 'Hawaii'])]
covid_categories = ['Admin2', 'Province_State', 'Lat', 'Long_', 'Confirmed', 'Deaths', 'Recovered']
covid_data = covid_data[covid_categories]
print(population_data.shape, covid_data.shape)

merged_data = pd.merge(covid_data, population_data, how='left', left_on=['Admin2', 'Province_State'], right_on=['County', 'State'])

print('shape', merged_data.shape)

categories = ['Lat', 'Long_', 'Confirmed', 'TotalPop', 'Transit', 'WorkAtHome']
nan_indexes = []
for category in categories:
    nan_indexes.extend(np.where(np.isnan(merged_data[category]))[0])
nan_indexes = np.asarray(nan_indexes)
nan_indexes = np.unique(nan_indexes)
merged_data = merged_data.drop(merged_data.index[nan_indexes.tolist()])
merged_data = merged_data[merged_data["Confirmed"] > 10]
print('shape', merged_data.shape)

u = merged_data['Lat']
v = merged_data['Long_']
coords = list(zip(u, v))
coords = np.asanyarray(coords)

y = merged_data['Confirmed'].values.reshape((-1, 1))
merged_data.to_csv(path + "/merged_data.csv")
merged_data = merged_data.drop(columns=['County', 'State', 'Admin2', 'Province_State', 'Deaths', 'Recovered',
                                        'Confirmed', 'Lat', 'Long_'])

x = merged_data.values


store(x, y, coords, path)

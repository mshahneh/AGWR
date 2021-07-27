#%%
import geopandas as gpd
import json
from bokeh.io import show
from bokeh.models import (CDSView, ColorBar, ColumnDataSource,
                          CustomJS, CustomJSFilter,
                          GeoJSONDataSource, HoverTool,
                          LinearColorMapper, Slider)
from bokeh.layouts import column, row, widgetbox
from bokeh.palettes import brewer
from bokeh.plotting import figure
from bokeh.palettes import Spectral4
import pandas as pd
import numpy as np
import pickle
import os

dataset = "kingHousePrices"
path = os.path.dirname(os.path.abspath(__file__ + str("/../../"))) + "/Data/" + dataset + "/"
merged_data = pd.read_csv(path + "data.csv")
with open(path + 'training_idx.data', 'rb') as filehandle:
    training_idx = pickle.load(filehandle)
with open(path + 'validation_idx.data', 'rb') as filehandle:
    validation_idx = pickle.load(filehandle)
with open(path + 'test_idx.data', 'rb') as filehandle:
    test_idx = pickle.load(filehandle)
with open(path + 'y.data', 'rb') as filehandle:
    y = pickle.load(filehandle)

with open(path + 'predictAll.data', 'rb') as filehandle:
    predictAll = pickle.load(filehandle)
predictAll = predictAll.reshape(-1, 1)
if np.mean(predictAll) > 10:
    predictAll = np.round(predictAll)
y = y.reshape(-1, 1)
predicterror = np.abs(predictAll-y)/y
print(type(predicterror), predicterror.shape)
size = predicterror*10+4


# ratio = np.zeros(len(merged_data))
# for i in range(len(merged_data)):
#     ratio[i] = float(merged_data.iloc[i]['Confirmed'])/float(merged_data.iloc[i]['TotalPop'])
#
#
# Min = min(ratio)
# Max = max(ratio)
# size = ((ratio*100)/3)+4
# ratio = ratio*100
# merged_data = merged_data.join(pd.DataFrame(data=ratio, columns=["ratio"]))
merged_data = merged_data.join(pd.DataFrame(data=size, columns=["size"]))
merged_data = merged_data.join(pd.DataFrame(data=predicterror, columns=["prediction_error"]))
merged_data = merged_data.join(pd.DataFrame(data=predictAll, columns=["predictAll"]))
merged_data = merged_data.join(pd.DataFrame(data=list(range(len(merged_data))), columns=["index"]))
print(merged_data.keys())
if "Unnamed: 0" in merged_data.keys():
    merged_data = merged_data.drop(columns="Unnamed: 0")

tooltips = [(x, '@'+x) for x in merged_data.keys()]
hover = HoverTool(names=["foo", "bar"])
p = figure(title=dataset,
           plot_height=600,
           plot_width=950,
           toolbar_location='below',
           tools=[hover, "pan, wheel_zoom, box_zoom, reset"],
           tooltips=tooltips)

p.xgrid.grid_line_color = None
p.ygrid.grid_line_color = None


for index, name, color in zip([training_idx, test_idx, validation_idx], ["Train", "Test", "Validation"], ['#440154', '#FFEB3B', '#35B778']):
    p.circle(x="long", y="lat", size='size', fill_color=color, fill_alpha=1, name="foo",
             source=merged_data.iloc[index], legend_label=name)

p.legend.location = "bottom_left"
p.legend.click_policy = "hide"

show(p)

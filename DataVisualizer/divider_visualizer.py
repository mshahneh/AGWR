#%%

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import matplotlib.cm as cm
import os

#Reading the data and creating the arrays
#%%

# dataset = "NYCAirBnb"
dataset = "kingHousePrices"
path = os.path.dirname(os.path.abspath(__file__ + str("/../../"))) + "/Data/" + dataset + "/"
with open(path + 'x.data', 'rb') as filehandle:
    x = pickle.load(filehandle)
with open(path + 'y.data', 'rb') as filehandle:
    y = pickle.load(filehandle)
with open(path + 'coords.data', 'rb') as filehandle:
    coords = pickle.load(filehandle)

#colors and shapes

#%%
# colors = ["b", "g", "r", "c", "m", "y", "k", "g"]
# NUM_COLORS = 10
# cm = plt.get_cmap('gist_rainbow')
# colors = [cm(1.*i/NUM_COLORS) for i in range(NUM_COLORS)]
colors = list(cm.rainbow(np.linspace(0, 1, 8)))
import random
random.shuffle(colors)
shapes = ["o", "<", "s", "*", "h", "x", "D", "h"]


from src.DataDivider import Divider
sections = 8
method_names = ["equalCount", "equalWidth", "kmeans"]


def display(divider, method_name):
    indices = divider.section_indices
    minx = 1000
    miny = 1000
    maxx = -1000
    maxy = -1000
    for i in range(len(indices)):
        temp = coords[result[i]]
        minx = min(minx, min(temp[:, 0]))
        maxx = max(maxx, max(temp[:, 0]))
        miny = min(miny, min(temp[:, 1]))
        maxy = max(maxy, max(temp[:, 1]))
        plt.scatter(temp[:, 0], temp[:, 1], c=colors[i], s = 2, marker=shapes[i%8])

    if method_name == "equalCount" or method_name == "equalWidth":
        print(method_name)
        for i in range(len(indices)):
            print(len(indices[i]))
        c = 0
        for boundary in divider.settings["boundaries"]:
            c += 1
            for j in range(2):
                if j == 0:
                    if c > divider.sections[1]:
                        continue
                    x1 = boundary[0][0]
                    x2 = boundary[1][0]
                    y1 = boundary[1][1]
                    y2 = boundary[1][1]
                elif j == 1:
                    if c % divider.sections[1] == 0:
                        continue
                    x1 = boundary[1][0]
                    x2 = boundary[1][0]
                    y1 = boundary[0][1]
                    y2 = boundary[1][1]
                # elif j == 2:
                #     x1 = boundary[0][0]
                #     x2 = boundary[1][0]
                #     y1 = boundary[0][1]
                #     y2 = boundary[0][1]
                # else:
                #     x1 = boundary[0][0]
                #     x2 = boundary[0][0]
                #     y1 = boundary[0][1]
                #     y2 = boundary[1][1]
                if x1 < minx:
                    x1 = minx
                if x1 > maxx:
                    x1 = maxx

                if x2 < minx:
                    x2 = minx
                if x2 > maxx:
                    x2 = maxx

                if y1 < miny:
                    y1 = miny
                if y1 > maxy:
                    y1 = maxy

                if y2 < miny:
                    y2 = miny
                if y2 > maxy:
                    y2 = maxy
                
                plt.plot([x1, x2], [y1, y2], color='0.4', linestyle='dashed')
                print(x1, x2, y1, y2)
    plt.axis('off')
    # plt.tick_params(top='off', bottom='off', left='off', right='off', labelleft='off', labelbottom='on')
    plt.savefig(method_name + '.png', bbox_inches='tight', pad_inches=0, dpi=500)
    plt.show()




for method_name in method_names:
    if method_name == "equalCount" or method_name == "equalWidth":
        divider = Divider([2, 4], method_name)
    else:
        divider = Divider([8], method_name)

    result = divider.divide(coords)
    display(divider, method_name)

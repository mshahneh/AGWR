import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os

path = path = os.path.dirname(os.path.abspath(
    __file__ + str("/.."))) + "/experiments/Results/divide_size.data"
results_location = os.path.dirname(os.path.abspath(__file__)) + "/"

with open(path, 'rb') as filehandle:
    experiment_result = pickle.load(filehandle)

datasets = list(experiment_result.keys())
shortDatasets = {'kingHousePrices': 'KGP', 'syntheticData1':'SD1'}
print(shortDatasets[datasets[0]])

pipe_line_time = {}
pipe_line_accuracy = {}
pipe_line_x_values = {}

ens_time = {}
ens_accuracy = {}
ens_x_values = {}


for i in range(len(datasets)):
    methods = list(experiment_result[datasets[i]].keys())
    l = 5
    pipe_line_x_values[datasets[i]] = []
    pipe_line_time[datasets[i]] = []
    pipe_line_accuracy[datasets[i]] = np.zeros((l, l))
    for j in range(0, len(methods), 2):
        # print(methods[j].split(" ")[-1])
        name = methods[j].split(" ")[0]
        # print(name)
        temp = methods[j].split(" ")[-1]
        a = int(temp.split("*")[0])
        b = int(temp.split("*")[1])
        pipe_line_time[datasets[i]].append(experiment_result[datasets[i]][methods[j]]["time"])
        pipe_line_accuracy[datasets[i]][a-1][b-1] = experiment_result[datasets[i]][methods[j]]["test_R2"]
        pipe_line_x_values[datasets[i]].append(a*b)

for i in range(len(datasets)):
    methods = list(experiment_result[datasets[i]].keys())
    l = 5
    ens_x_values[datasets[i]] = []
    ens_time[datasets[i]] = []
    ens_accuracy[datasets[i]] = np.zeros((l, l))
    for j in range(1, len(methods), 2):
        # print(methods[j].split(" ")[-1])
        name = methods[j].split(" ")[0]
        # print(name)
        temp = methods[j].split(" ")[-1]
        a = int(temp.split("*")[0])
        b = int(temp.split("*")[1])
        # print(a, b, experiment_result[datasets[i]][methods[j]]["time"])
        ens_time[datasets[i]].append(experiment_result[datasets[i]][methods[j]]["time"])
        ens_accuracy[datasets[i]][a-1][b-1] = experiment_result[datasets[i]][methods[j]]["test_R2"]
        ens_x_values[datasets[i]].append(a*b)


y1_pipe_line = [v for _,v in sorted(zip(pipe_line_x_values[datasets[0]], pipe_line_time[datasets[0]]))] 
y2_pipe_line = [v for _,v in sorted(zip(pipe_line_x_values[datasets[i]], pipe_line_time[datasets[1]]))]
x_pipe_line = sorted(pipe_line_x_values[datasets[i]])
indexes = [x_pipe_line.index(_) for _ in set(x_pipe_line)]
x_pipe_line = [x_pipe_line[_] for _ in indexes]
y1_pipe_line = [y1_pipe_line[_] for _ in indexes]
y2_pipe_line = [y2_pipe_line[_] for _ in indexes]
print(y1_pipe_line)


y1_ens = [v for _,v in sorted(zip(ens_x_values[datasets[0]], ens_time[datasets[0]]))] 
y2_ens = [v for _,v in sorted(zip(ens_x_values[datasets[i]], ens_time[datasets[1]]))]
x_ens = sorted(ens_x_values[datasets[i]])
indexes = [x_ens.index(_) for _ in set(x_ens)]
x_ens = [x_ens[_] for _ in indexes]
y1_ens = [y1_ens[_] for _ in indexes]
y2_ens = [y2_ens[_] for _ in indexes]

plt.rc('font', size=25)
plt.rc('axes', titlesize=25)
plt.rc('axes', labelsize=25)
fig = plt.figure()
ax1 = fig.add_subplot(111)
ax2 = ax1.twinx()
ax1.plot(x_pipe_line, y1_pipe_line, '-b', label="Pipeline "+shortDatasets[datasets[0]])
ax2.plot(x_pipe_line, y2_pipe_line, 'b--', label="Pipeline "+shortDatasets[datasets[1]])
ax1.plot(x_ens, y1_ens, '-r', label="Ensemble "+shortDatasets[datasets[0]])
ax2.plot(x_ens, y2_ens, 'r--', label="Ensemble "+shortDatasets[datasets[1]])

# yint = range(round(min(min(y1_pipe_line), min(y1_ens))), round(max(max(y1_pipe_line), max(y1_ens))))

# ax.set_yticks(yint)
fig.legend(loc="upper right", bbox_to_anchor=(1,1), bbox_transform=ax1.transAxes, prop={"size":22})


leftLablel = shortDatasets[datasets[0]] + " time"
rightLable = shortDatasets[datasets[1]] + " time"
ax1.set_xlabel("Number of sections")
ax1.set_ylabel(leftLablel)
ax2.set_ylabel(rightLable)

plt.savefig(results_location + 'divide_size_time_chart.png', bbox_inches='tight', pad_inches=0, dpi=500)
plt.show()
plt.cla()
plt.clf()
plt.close()


fig, axes = plt.subplots(nrows=2, ncols=2)
fig.tight_layout() 
j = 0
for row in axes:
    i = 0
    for col in row:
        if j == 0:
            x = np.asarray(list(range(1, 7)))
            y = np.asarray(list(range(1, 7)))
            z = pipe_line_accuracy[datasets[i]]
            title = "Pipeline " + shortDatasets[datasets[i]]
        else:
            x = np.asarray(list(range(1, 7)))
            y = np.asarray(list(range(1, 7)))
            z = ens_accuracy[datasets[i]]
            title = "Ensemble " + shortDatasets[datasets[i]]
        ax = col
        c = ax.pcolormesh(x, y, z, cmap='YlGnBu', vmin=z.min()*0.9, vmax=z.max()*1.1)
        ax.set_title(title)
        fig.colorbar(c, ax=ax)
        ticks = [1.5, 2.5, 3.5, 4.5, 5.5]
        ax.set_yticks(ticks)
        ax.set_yticklabels(list(range(1, 6)))
        ax.set_xticks(ticks)
        ax.set_xticklabels(list(range(1, 6)))

        i += 1
    j+=1

# plt.axis('off')
# plt.tick_params(top='off', bottom='off', left='off', right='off', labelleft='off', labelbottom='on')
plt.savefig(results_location + 'divide_size_accuracy_charts.png', bbox_inches='tight', pad_inches=0, dpi=500)

plt.show()
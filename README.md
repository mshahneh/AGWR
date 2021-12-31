# A-GWR

Implimentation of A-GWR modular framework and S-MGWR spatial model.

# Table of Contents

* [How to install](#how-to-install)
    + Libraries
    + MGWR and GWR
* [Quick Start](#quick-start)
* [How does it work](#How-does-it-work)
    + What is the Modular framework
      - Pipeline
      - Ensemble
    + What is S-MGWR
* [Project Breakdown](#project-breakdown)
    + Modular Framework
      - DataDivider
      - ModularFramework
    + S-MGWR
      - SpaceSearch
      - Cache
* [Results](#Results)

# How to Install

## Libraries

The libraries used in this project along with the version A-GWR was tested on are as follows

```
```

## MGWR and GWR

We used a modified version of MGWR and GWR in our code. The modified versions are included here. For more information on
GWR code check (link fix)

# Quick Start

For more information check the example file in the root directory. Both spatial and ml modules need to have a train and
a predict function. A few samples for both spatial and ml modules are listed in the ```helper/SampleModules```
directory.

To run the example after navigating to the project location, run:

```
python -m example
```

The example code, uses a ```SMGWR``` model as spatial module and ```Random Forest``` as the general purpose ml module.

# How does it work

A-GWR contains two parts, a novel spatial module called S-MGWR, and a framework to combine spatial and general purpose
ML models.

## What is the Modular framework

We introduce a modular framework to combine spatial models and general purpose models. In traditional methods, such as
ordinary least squares, the role of location is often neglected or is not properly captured. Many spatial models such as
GWR are introduced to overcome this issue. However, these methods often cannot capture non-linear relations. In A-GWR
modular framework, by combining spatial models and general-purpose ml models, we have tried to solve this problem.

We have two methods to combine the modules, the Pipeline (spatial modelfirst) method and the Ensemble (general-purpose
model first) method.

### Pipeline

The following figure shows how the pipeline method works.
![pipeline figure](Images/new_pipeline_color.PNG?raw=true)

### Ensemble

The following figure shows how the Ensemble method works.
![ensemble figure](Images/new_ens_color.PNG?raw=true)

## What is S-MGWR

S-MGWR is a stateless variation of MGWR spatial model that assigns different localities to different features. S-MGWR
extends MGWR by eliminating the need for a history of bandwidth values.

In S-MGWR, the search for bandwidths can be treated like a blackbox optimization and can also be done in parallel.

In our implementation, to speed up the training process, we've also implemented an LRU cache system to keep the recently
computed weight values.

# Project Breakdown

In this section, we break down the project in more details.

## Data

The ``` Data ``` directory, holds the dataset we originally used. In ``` Data/DataHandlers``` the codes we used to clean
or create each dataset is provided.

## helpers

In the ```helpers/SampleModules``` directory contains a few samples of spatial and ML modules. The Spatial Module
contains ``` GWR, MGWR, and S-MGWR```. The S-MGWR is configured to
perform ``` Successive Halving, SPSA, Bayesian Optimization, and Hill Climbing``` but can be changed to perform any
space search method suitable for the user's application.

## Modular Framework

Both pipeline and ensemble methods divide the data into sections. This is done by the data divider code. The modular
framework creates an instance of the datadivider and combines it with the target architecture (pipeline or ensemble)

### DataDivider

The code for Random, k-means, grid divide is implemented in DataDivider. In addition, for the test data, in order to
find the influence of each section on the test data point, the ``` weighted_mean ``` function is implemented.

### ModularFramework

This class has the implementation of the modular framework. In the initialization, the spatial model, general purpose
model, and the configuration (such as the divide method) is passed to this class.

The ``` train ``` function, based on the method (pipeline, ensemble) creates the threads and runs the appropriate
modules.

## S-MGWR

This directory contains the implementation codes for the S-MGWR spatial model.

### SpaceSearch

To find the best set of bandwidths, some of the most famous hyper parameter optimization algorithms have been
implemented. These methods implemented are:

```
1- Hyperband
2- Successive Halving
3- Simulated Annealing
4- Hill Climbing
5- Golden Section Section
6- Bayesian Optimization
7- Simultaneous perturbation stochastic approximation (SPSA)
```

### Cache

We have also implemented an LRU cache system to store the weights computed for recently used bandwidths



# Results
|               | pGeorgia         | pTokyo          | pClrwatr        | pBerlin         | synthData1      | synthData2       | kingHousePrices  | NYCAirBnb        |
|---------------|------------------|-----------------|-----------------|-----------------|-----------------|------------------|------------------|------------------|
|               | 1-R<sup>2</sup>  | 1-R<sup>2</sup> | 1-R<sup>2</sup> | 1-R<sup>2</sup> | 1-R<sup>2</sup> | 1-R<sup>2</sup>  | 1-R<sup>2</sup>  | 1-R<sup>2</sup>  |
| A-GWR         | 0.4161 ± 0.05    | 0.0164 ± 0.0017 | 0.9226 ± 0.0645 | 0.6821 ± 0.0145 | 0.0137 ± 0.0008 | 0.061 ± 0.0068   | 0.1109  | 0.4181           |
| MGWR          | 0.4913 ± 0.0689  | 0.0226 ± 0.0038 | 0.9622 ± 0.0713 | 0.6965 ± 0.0039 | 0.0139 ± 0.0014 | 0.0999 ±  0.009  |          0.1341        |       0.4378           |
| GWR           | 0.3914 ± 0.0629  | 0.0215 ± 0.0032 | 0.9265 ± 0.0577 | 0.6876 ± 0.0076 | 0.0805 ± 0.0108 | 0.1168 ± 0.0068  | 0.1295   |      0.4284            |
| Random Forest | 0.3878 ± 0.0346  | 0.0352 ± 0.0024 | 0.9276 ± 0.0589 | 0.5825 ± 0.0094 | 0.1788 ± 0.0041 | 0.1240 ± 0.0019  | 0.1216   | 0.3927  |
| XGBoost       | 0.6229 ± 0.0832  | 0.0359 ± 0.0026 | 1.0304 ± 0.0561 | 0.6942 ± 0.0182 | 0.1634 ± 0.0051 | 0.1042 ±  0.0008 | 0.1165   | 0.4092  |

# Code for ["A mechanically-inspired geometric model to predict microbial growth across environments"](https://doi.org/10.1101/2025.07.02.662826)

## Overview

[Publication](#publication)

[Brief description of Voter Model](#brief-description-of-voter-model)

[Directory structure and executing code](#directory-structure-and-executing-code)



## Publications

Manuscript awaiting publication.

Data derived from and fitting employed [here](https://doi.org/10.1101/2025.04.30.651255)

We have included all data files and results pertinent to the manuscript (except those larger than 100MB).

## Brief description of the geometric model

The geometric model of bacterial growth models how the concentration of bacteria, $X(t)$, as the proportion of inactive components within the cell, $I(t)$ varies.

$$\frac{dX}{dt} = \alpha e^{-\beta I(t)}$$

$$
I(t)=
\begin{cases}
I_0e^{-\gamma_1 t} & \quad t\lt T\\
1 + (I_0 e^{-\gamma_1 T} - 1)e^{-\gamma_2 (t-T)} & \quad t\ge T
\end{cases}
$$

Where:

- $\alpha$ is the maximum growth rate of bacteria
- $\beta$ is the exponent relating inactivity of components to decay of cellular growth
- $\gamma_1$ is the rate of component activation during the lag phase
- $\gamma_2$ is the rate of component inactivation during the saturation pahse
- $T$ is the time at which the exponential phase transitions to the saturation phase
- $I_0$ is the initial proportion of inactive components.


## Directory Structure and Executing Code
```
.
├── Bash_Copying.sh
├── PaperFigs
├── RawFigures
└── Simulations
    ├── Core
    ├── Deprecated
    ├── Fig1
    ├── Fig2
    ├── Fig3_Coculture
    └── __pycache__
```

### PaperFigs
.afpub and corresponding .pdf files for the manuscript.

### RawFigures
Source linking the .png files to the .afpub files

### Core
Files containing ubiquitous functions and data

- `Data.py`: The data obtained from [here](https://doi.org/10.1101/2025.04.30.651255)
- `Models.py`: The geometric model (for different contexts) and other fittings, such as the logistic, Gompertz and kinetic models.
- `Objectives.py`: The objective functions to be minimised for least-square fittings

### Simulations
Each figure can be generated with `python Script.py`.

#### Fig1
Parameter fitting of *S. aureus* in closed-system monoculture, and open-system monoculture and coculture, at the same time.

Parameter fitting of *P. aeruginosa* in closed-system monoculture and open-system coculture.

Save the fitting results in a `.npz` file for use in Fig2 and Fig3

#### Fig2
Parameter fitting for different fits of monoculture data for logistic, Gompertz and a kinetic model, for the sake of comparison to the geometric model.

#### Fig3_Coculture
Parameter fitting for closed-system coculture, adding a killing chemical secreted by *P. aeruginosa* which kills *S. aureus*. Fitting is used in [here](https://doi.org/10.1101/2025.04.30.651255).


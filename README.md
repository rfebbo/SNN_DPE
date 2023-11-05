# Installation:
cd src

pip install --editable .

# Introduction

This repository contains code for generating random spiking neural networks as reservoirs. These reservoirs are simulated for some time with spiking inputs relative to a datasample in a data set. Then, with a matrix of weights(DPE) which uses the average spiking rate of each neuron as the input, gradient descent is used to train the weights on classificaion and timeseries prediction tasks.

# Step-by-step outline with links to demonstration notebooks

* [Training / EO idea](./demonstration/06_training_test.ipynb)
    1. [Generate a random SNN](./demonstration/02_network_creation.ipynb)
    1. for each sample in the dataset
        - [feed sample into SNN with encoders and simulate for some time (sim_time)](./demonstration/03_running_network.ipynb)
        - [find steady state](./demonstration/04_steady_state.ipynb)
        - [find the average neuron fire rates over the steady state (x) and do Mat Mul with DPE weights (y) (forward_pass)](./src/training_tools.py)
        - [train DPE layer with Gradient Descent](./demonstration/05_weight_update.ipynb)
    1. Final accuracy (trainability of SNN) can be used as the fitness of generated SNN
    1. number of output neurons required can also be used as the fitness of generated SNN (would require multiple DPEs per SNN to be tested (should be fast))
    1. [Save SNN with fitness](./demonstration/07_saving_network.ipynb)
    1. repeat 'till a collecion (population) of SNNs with fitnesses are obtained and do EO


# Extra

- Future Ideas
    1. Generate multiple SNNs and have one DPE for each of them
    1. train the DPEs like above, and use the collection of SNNs as a genome for EO
    1. Use multiple DPE, SNN outputs as lists of guesses like in Thousand Brains
    1. use single encoder with attributes passed in over time
    1. investigate unsupervised approach

- Notes
    1. In unshuffled datasets, the accuracy is artificially high during training. This is due to the DPE learning weights that can switch between classes after a few datasamples into the new class
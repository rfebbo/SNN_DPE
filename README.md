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

- Future Idea
    1. Generate multiple SNNs and have one DPE for each of them
    1. train the DPEs like above, and use the collection of SNNs as a genome for EO
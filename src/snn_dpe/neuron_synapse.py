import numpy as np

class Neuron:
    def __init__(self, id, threshold, leak):
        self.id = id
        self.membrane_voltage = 0
        self.threshold = threshold
        self.leak = leak
        self.synapses = []

    # accumulate then spike
    def update(self):
        if self.membrane_voltage < 0:
            self.membrane_voltage = 0
    
        if self.membrane_voltage > self.threshold:
            self.spike()
            self.membrane_voltage = 0
            
            return 1
        else:
            self.membrane_voltage -= self.leak
            return 0

    def spike(self):
        for s in self.synapses:
            s.send_spike()

    def apply_potential(self, potential):
        self.membrane_voltage += potential

    def add_synapse(self, synapse):
        self.synapses.append(synapse)

    def reset(self):
        self.membrane_voltage = 0

class Synapse:
    def __init__(self, n1, n2, weight, std_dev = 0, drift = 0):
        self.n1 = n1
        self.n2 = n2
        self.weight = weight
        self.std_dev = std_dev
        self.drift = drift

    def send_spike(self):
        self.weight += self.weight*self.drift
        self.n2.apply_potential(np.random.normal(self.weight, self.std_dev))
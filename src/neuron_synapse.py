class Neuron:
    def __init__(self, id, threshold, leak):
        self.id = id
        self.membrane_voltage = 0
        self.threshold = threshold
        self.leak = leak
        self.incoming_potential = 0
        self.synapses = []

    # accumulate then spike
    def update(self):
        self.membrane_voltage += self.incoming_potential
        
        if self.membrane_voltage < 0:
            self.membrane_voltage = 0
        
        self.incoming_potential = 0

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
        self.incoming_potential += potential

    def add_synapse(self, synapse):
        self.synapses.append(synapse)

    def reset(self):
        self.incoming_potential = 0
        self.membrane_voltage = 0

class Synapse:
    def __init__(self, n1, n2, weight):
        self.n1 = n1
        self.n2 = n2
        self.weight = weight

    def send_spike(self):
        self.n2.apply_potential(self.weight)
import numpy as np


class Neuron:
    def __init__(self, id, threshold, leak, spike_event=0):
        self.id = id
        self.membrane_voltage = 0
        self.threshold = threshold
        self.leak = leak
        self.spike_event = spike_event
        self.synapses = []

    # accumulate then spike
    def update(self):
        if self.membrane_voltage < 0:
            self.membrane_voltage = 0
            self.spike_event = 0
        for s in self.synapses:
            s.update()
        if self.membrane_voltage >= self.threshold:
            self.spike()
            self.membrane_voltage = 0
            self.spike_event = 1

            return 1
        else:
            self.membrane_voltage -= self.leak
            self.spike_event = 0
            return 0
        

    def spike_event_check(self):
        return self.spike_event


    def spike(self):
        for s in self.synapses:
            s.send_spike()

    def apply_potential(self, potential):
        self.membrane_voltage += potential

    def add_synapse(self, synapse):
        self.synapses.append(synapse)

    def reset(self, reset_synapses=True):
        self.membrane_voltage = 0

        if reset_synapses:
            for s in self.synapses:
                s.weight = s.start_weight

class Synapse:
    def __init__(self, n1, n2, weight, std_dev = 0, drift = 0, stdp=False):
        self.n1 = n1
        self.n2 = n2
        self.start_weight = weight
        self.weight = weight
        self.std_dev = std_dev
        self.drift = drift
        self.stdp = stdp
        self.stdp_mult = 0
        self.t_n1 = 0
        self.t_n2 = 0

    def send_spike(self):
        self.n2.apply_potential(np.random.normal(self.weight, self.std_dev))
        self.weight -= self.weight*self.drift       # drfit moves the weight closer to 0?
        


# STDP LUT
# time gap n2-n1     -3   -2     -1   0  1    2    3
# weight multiplier 
    def update(self):
        n1_spike_event = self.n1.spike_event_check()
        n2_spike_event = self.n2.spike_event_check()
        applied_mult = self.stdp_mult
        if n1_spike_event == 1:
            self.t_n1 = 1
            applied_mult = 0
        elif(self.t_n1 > 1):
            self.t_n1 += 1
        if n2_spike_event == 1:
            self.t_n2 = 1
            applied_mult = 0
        elif(self.t_n2 > 1):
            self.t_n2 += 1

        match (self.t_n1 - self.t_n2):  #is this correct polarity?
            case -3:
                self.stdp_mult = -0.001
            case -2:
                self.stdp_mult = -0.0025
            case -1:
                self.stdp_mult = -0.015
            case 0:
                self.stdp_mult = 1
            case 1:
                self.stdp_mult = 0.015
            case 2:
                self.stdp_mult = 0.0025
            case 3:
                self.stdp_mult = 0.001

        if self.stdp:
            #if ((self.t_n1 - self.t_n2) > -4) and ((self.t_n1 - self.t_n2) < 4):
            if(self.t_n1 > 0 and self.t_n2 > 0 and self.t_n1 < 5 and self.t_n2 < 5):
                if(applied_mult != self.stdp_mult):
                    self.weight += self.weight * self.stdp_mult
                    applied_mult = self.stdp_mult
        #else:
            
        
        if (self.weight > 10000): self.weight = 10000
        if (self.weight < -10000): self.weight = -10000
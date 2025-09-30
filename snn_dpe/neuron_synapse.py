import uuid

class Neuron:
    def __init__(self, threshold, leak):
        self.id = uuid.uuid4()
        self.threshold = threshold
        self.leak = leak

        self.membrane_voltage = 0
        self.synapses = []

    # accumulate then spike
    def update(self):
        spiked = False
        if self.membrane_voltage >= self.threshold:
            self.spike()
            self.membrane_voltage = 0

            spiked = True
        else:
            self.membrane_voltage -= self.leak
            if self.membrane_voltage < 0:
                self.membrane_voltage = 0

        for s in self.synapses:
            s.update()
        
        return spiked

    def spike(self):
        for s in self.synapses:
            s.queue_spike()

    def apply_potential(self, potential):
        self.membrane_voltage += potential

    def add_synapse(self, synapse):
        self.synapses.append(synapse)

    def reset(self):
        self.membrane_voltage = 0

class Synapse:
    def __init__(self, n1, n2, weight, delay : int):
        self.n1 = n1
        self.n2 = n2
        self.delay = delay
        self.weight = weight

        # flag for if there is an outgoing spike waiting
        self.spike_queued = False
        # update tick counter for when to send the spike based on delay
        self.update_count = 0

    def queue_spike(self):
        self.spike_queued = True

    def update(self):
        if self.spike_queued == False:
            return
        
        # check if the delay requirement has been met
        if self.update_count >= self.delay:
            self.n2.apply_potential(self.weight)
            self.spike_queued = False
            self.update_count = 0

        self.update_count += 1
        
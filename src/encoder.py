# rate encoder
#   higher values into the network -> more active network (since positive wieght values only)
#   lower values into the network -> less active network
class Encoder:
    def __init__(self, min_f, max_f, sim_f):
        self.last_fire = 0

        # min, max, and simulation frequencies that can be represented
        self.min_f = min_f
        self.max_f = max_f
        self.sim_f = sim_f

        
    def set_value(self, value):
        # using the min and max frequency, calculate the frequency of the value
        self.value = value
        self.value_f = ((self.max_f - self.min_f) * value) + self.min_f

        # using the simulation frequency, calculate the period of the value
        #   (the number of simulation time steps between fires)
        self.fire_period = self.sim_f / self.value_f
        
        # a potentialy faster, but memory hungry way to implement would be a look up table

    def update(self):
        self.last_fire += 1

        # fire everytime the fire period is completed
        if self.last_fire > self.fire_period:
            self.last_fire = 0
            return 1
        else:
            return 0

    def reset(self):
        self.last_fire = 0


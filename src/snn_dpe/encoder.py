# rate / period encoder
#   higher values into the encoder -> faster encoder spiking -> more active network (since positive wieght values only)
#   lower values into the encoder -> slower encoder spiking -> less active network
#   encoding the period directly forces the resolution across the input space to be the same, otherwise it will be squished at higher values
class Encoder:
    def __init__(self, min_f, max_f, sim_f, enc_type='period'):
        self.last_fire = 0

        # min, max frequencies that can be output by this encoder
        self.min_f = min_f
        self.max_f = max_f

        # the frequency at which the simulation should be running
        self.sim_f = sim_f

        self.enc_type = enc_type

    # set this to whatever value the encoder should be representing [0, 1]
    def set_value(self, value):
        self.last_fire = 0
        
        # using the min and max frequency, calculate the frequency of value
        self.value = value
        self.value_f = ((self.max_f - self.min_f) * value) + self.min_f

        # using the simulation frequency, calculate the period of the value
        #   (the number of simulation timesteps between fires)
        if self.enc_type == 'frequency':
            self.fire_period = int(self.sim_f / self.value_f)
        elif self.enc_type == 'period':
            self.fire_period = -(((self.sim_f/self.min_f - self.sim_f/self.max_f) * value) + self.sim_f/self.max_f) + (self.sim_f/self.min_f + self.sim_f/self.max_f)
            self.fire_period = int(self.fire_period)
        else:
            print('error')

    def update(self):
        # step one timestep into simulation (if sim_f is 1000 Hz, one timestep is 1ms)
        self.last_fire += 1

        # fire everytime the fire period is completed
        if self.last_fire > self.fire_period:
            self.last_fire = 0
            return 1
        else:
            return 0

    def reset(self):
        self.last_fire = 0


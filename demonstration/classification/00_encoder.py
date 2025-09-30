import matplotlib.pyplot as plt
import numpy as np

from snn_dpe import Encoder

# output frequency range of the spike encoders
min_f = 10
max_f = 100

# simulation frequency (encoder update rate)
sim_f = 1000

e1 = Encoder(min_f, max_f, sim_f) 
e2 = Encoder(min_f, max_f, sim_f)
e3 = Encoder(min_f, max_f, sim_f)

value1 = 0.0
value2 = 0.1
value3 = 1.0

e1.set_value(value1)
e2.set_value(value2)
e3.set_value(value3)

# simulate for one second
fires1 = []
fires2 = []
fires3 = []
for t in range(sim_f):
    if e1.update():
        fires1.append(t)

    if e2.update():
        fires2.append(t)

    if e3.update():
        fires3.append(t)


fig = plt.subplots(figsize=(13,2))
plt.title(f'Encoder Outputs with an output frequency range of {min_f} Hz - {max_f} Hz\
          \n Simulated at {sim_f} Hz for 1 second')
e_labels=[
        f'e1: {value1} -> {e1.value_f} Hz',\
        f'e2: {value2} -> {e2.value_f} Hz',\
        f'e3: {value3} -> {e3.value_f} Hz']
plt.yticks(ticks=[1, 2, 3], labels=e_labels)
plt.ylabel('Encoders')
plt.ylim(0.5, 3.5)
plt.xlabel('Time (ms)')
plt.xticks(ticks=range(0, sim_f + 1, 100))

plt.scatter(fires1, np.ones(len(fires1)) * 1, marker='|')
plt.scatter(fires2, np.ones(len(fires2)) * 2, marker='|')
plt.scatter(fires3, np.ones(len(fires3)) * 3, marker='|')

plt.show()
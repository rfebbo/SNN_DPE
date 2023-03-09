import matplotlib.pyplot as plt
import numpy as np

from snn_dpe.tools.network import (reset_network, run_network_early_exit,
                                   run_network_timeseries)
from snn_dpe.tools.train import forward_pass, mse

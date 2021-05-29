import numpy as nu
import pickle
from matplotlib import pyplot as pl


result = None
with open("SACAgent_tempResults.pickle", "rb") as file:
    result = pickle.load(file)

# return = steps, rewards, losses
pl.xlabel('Step', fontsize=22)
pl.ylabel('Returns', fontsize=22)
pl.tick_params(labelsize=16)
pl.plot(result[:, 0], result[:, 1])
pl.show()

import numpy as np
import pickle as pkl


cascades = [pkl.load(open('data/digg/cascade_{}.pkl'.format(i), 'rb'))
            for i in range(5)]
sizes = [len(np.nonzero(c >= 0)[0]) for c in cascades]
print(np.mean(sizes))
print(np.std(sizes))

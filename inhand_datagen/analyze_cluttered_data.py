import pickle
import matplotlib.pyplot as plt
import numpy as np

occlusions=np.array(pickle.load(open('/media/willie/fda8d90f-998c-425a-9823-a20479be9e98/data/cluttered_datasets/occlusions.p', 'rb')))

num_bins=1000
x_pos=np.arange(num_bins)
counts=np.zeros(num_bins)
for ind in range(x_pos.shape[0]):
    counts[ind]=np.sum(np.logical_and(occlusions>=ind/num_bins, occlusions<(ind+1)/num_bins))
counts=counts/occlusions.shape[0]
    
plt.bar(x_pos, counts, color='green')
plt.show()
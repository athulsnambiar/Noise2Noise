import sys
from dataloader import NoiseDataset
import matplotlib.pyplot as plt
import numpy as np

path = sys.argv[1]

dataset = NoiseDataset(path=path,
                       count=100,
                       noisetype="gaussian",
                       std=1)

n1, n2 = dataset[1]
# print(n1.shape)

plt.imshow(n1.astype(np.uint8))
plt.show()
plt.imshow(n2.astype(np.uint8))
plt.show()

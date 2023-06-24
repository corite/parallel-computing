#!/usr/bin/python3

import fileinput
import sys
import matplotlib.pyplot as plt
import numpy as np

# csv can be piped into this script ()
data = np.genfromtxt(sys.stdin, delimiter=',')
fig = plt.figure()
plt.xlabel(sys.argv[1])
plt.ylabel(sys.argv[2])

ax1 = fig.add_subplot(111)
ax1.plot(range(data.shape[0]), data[:, 0])
plt.show()

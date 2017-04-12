#!/usr/bin/env python
import numpy as np
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import pandas as pd

def plot_histogram(x, x2=None):
  # series 1
  # the histogram of the data
  n, bins, patches = plt.hist(x, 50, normed=1, facecolor='green', alpha=0.75)
  # add a 'best fit' line
  y = mlab.normpdf(bins, mu, sigma)
  l = plt.plot(bins, y, 'g--', linewidth=1)
  # series 2
  if not x2 == None:
    n, bins, patches = plt.hist(x2, 50, normed=1, facecolor='red', alpha=0.75)
    y = mlab.normpdf(bins, mu2, sigma2)
    l = plt.plot(bins, y, 'r--', linewidth=1)

  plt.xlabel('bin count')
  plt.ylabel('values')
  plt.title(r'Frequencygram of values')
  plt.axis([-1, 3, 0, 15])
  plt.grid(True)
  plt.show()

# data = pd.read_csv('/home/smorrell/Dropbox/Dream_scripts_etc/logs-local/20170409logits.csv')
# x = data.values[:, 0]
# mu, sigma = np.average(x), np.std(x)
# x2 = data.values[:, 1]
# mu2, sigma2 = np.average(x2), np.std(x2)
# plot_histogram(x)  # , x2)



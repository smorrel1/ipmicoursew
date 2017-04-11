#! /usr/bin/python
import numpy as N
import numpy as np
import os
import scipy.stats
from PIL import Image
from nifti import *
# import nibabel as nib
from pylab import *
from time import time

# Constants
path = '/home/smorrell/git/ipmi/MPHGB06_coursework_part'
path_in = path + '1/images/'
path_out_ff = path + '1/out_ff/'
path_out_priors = path + '1/out_priors/'  # transformed priors
path_seg = path + '1/seg/'                # segmented images

# VARIABLES
# pik is the probability of pixel i being in class k (like fuzzy sets) ndims=4, [x, y, z, k]
timepoint = time()
def bootlog(msg):
  global timepoint
  timegap = time() - timepoint
  timepoint = time()
  print(" * %s - %f " % (msg, timegap))

def read_file(filename):
  imagefile = NiftiImage(filename)
  image_array = imagefile.asarray()
  image_array = np.transpose(image_array)
  return image_array.astype('float32')

def save_file(array, filename):
  # coord_transform = np.eye(4)  # TODO: is the transform the same as np.transpose?
  array = np.transpose(array)
  nim = NiftiImage(array)
  nim.save(filename)

def uMRF (pik, k):
  """Calcualtes the Markov Random Field for the whole image for one class k of the central pixel
  Args: 
    pik: class probability per pixel for whole image with dims [x, y, z, numclass]
    k: the class of the central pixel. uMRF calculates the prior for this class. 
  For each pixel (outer 3 lops) we sum the class probs by neighbour & central class into a matrix then * G
  Returns: 
  G energy matrix. penalty for being near pixels. I.e. different = 1, same = 0.  
  Horizontal axis is class j of neighbour. Vertical axis is class k of central pixel."""
  G = np.ones([6, 6]) - np.eye(6)
  umrf = np.ndarray([np.size(pik, 0), np.size(pik, 1), np.size(pik, 2)])
  for x in range(0, np.size(pik, 0)):
    for y in range(0, np.size(pik, 1)):
      for z in range(0, np.size(pik, 2)):
        umrfAtPixel = 0
        for j in range(0, numclass):  # slide 4.53 outer sum over classes of neighbouring pixels
          umrfj = 0
          if x > 0:  # inner sum over neighbours
            umrfj += pik[x-1, y, z, j]
          if x+1 < np.size(pik, 0):
            umrfj += pik[x+1, y, z, j]
          if y > 0:
            umrfj += pik[x, y-1, z, j]
          if y+1 < np.size(pik, 1):
            umrfj += pik[x, y+1, z, j]
          if z > 0:
            umrfj += pik[x, y, z-1, j]
          if z+1 < np.size(pik, 2):
            umrfj += pik[x, y, z+1, j]
          umrfAtPixel += umrfj * G[k, j]
        umrf[x, y, z] = umrfAtPixel
  return umrf

# file_name = '1056_F_71.22_AD_60740.nii'
# file_name = '1221_M_71.27_AD_48975.nii'
for file_name in os.listdir(path_in):
  print "Loading image", file_name
  imgData = read_file(path_in + file_name)
  # Priors
  image_prior = read_file(path_out_priors + 'propagated_priors_' + file_name)  # dims:
  image_prior[image_prior == 0] = 10**-7

  didNotConverge = 1
  numclass = 4

  # Allocate space for the posteriors
  classProb = np.ndarray([np.size(imgData, 0), np.size(imgData, 1), np.size(imgData, 2), numclass])  # [x, y, z, 4]
  classProbSum = np.ndarray([np.size(imgData, 0), np.size(imgData, 1), np.size(imgData, 2)])  # [x, y, z]


  # initialise mean and variances
  bootlog('initialising')
  mean = np.zeros(numclass)
  var = np.zeros(numclass)
  for classIndex in range(0, numclass):
    pik = image_prior[:, :, :, classIndex]  # here pik just for one class, restored below
    mean[classIndex] = np.sum(pik*imgData)/np.sum(pik)  # pik is [4, 182, 218] and imgData is [182, 218, 182]
    var[classIndex] = np.sum(pik * ((imgData - mean[classIndex])**2)) / np.sum(pik)
  pik = image_prior
  # mean = np.reshape(np.mean(classPrior, (0, 1)), [4, 1]) * 256 + 100  # why does +100 increase convergence?
  # var = (np.random.rand(numclass, 1) * 10) + 200

  logLik = -1000000000
  oldLogLik = -1000000000

  # Initialise MRF
  beta = 0.5
  MRF = np.ones([np.size(imgData, 0), np.size(imgData, 1), np.size(imgData, 2), numclass])

  # Iterative process
  iteration = 0
  while didNotConverge:
    iteration += 1
    # bootlog('iteration ' + str(iteration))
    # Expectation
    classProbSum[:, :, :] = 0
    for classIndex in range(0, numclass):
      gaussPdf = scipy.stats.norm.pdf(imgData - mean[classIndex], scale=np.sqrt(var[classIndex])) * image_prior[:, :, :, classIndex]
      # = f(yi|zi=ek, phi) = G(yi-muk)
      gaussPdf2 = 1/np.sqrt(2*np.pi*var[classIndex])*np.exp(-(imgData-mean[classIndex])**2/(2*var[classIndex])) \
                  * image_prior[:, :, :, classIndex]
      assert abs(gaussPdf - gaussPdf2).max() < 10**-8  # implementation check
      # include factors for f(zi = ek): pi = image_prior and MRF = exp(-beta*Umrf()) / normalising term
      classProb[:, :, :, classIndex] = gaussPdf * image_prior[:, :, :, classIndex] * MRF[:, :, :, classIndex]  # slide 60
      classProbSum[:, :, :] += classProb[:, :, :, classIndex]
    classProbSum[classProbSum <= 0] = 10**-7

    # normalise posterior
    for classIndex in range(0, numclass):
      classProb[:, :, :, classIndex] = classProb[:, :, :, classIndex] / classProbSum[:, :, :]

    # Cost function
    oldLogLik = logLik
    logLik = np.sum(np.log(classProbSum))  # slide 47 equation 2

    # Maximization slide 46
    for classIndex in range(0, numclass):
      pik = classProb[:, :, :, classIndex]
      mean[classIndex] = np.sum(pik*imgData) / np.sum(pik)
      var[classIndex] = np.sum(pik * (imgData - mean[classIndex])**2) / np.sum(pik)
      MRF[:, :, :, classIndex] = np.exp(-beta * uMRF(classProb, classIndex))

      print 'for class ' + str(classIndex) + ', mean = ' + str(mean[classIndex]) + ', var = ' + str(var[classIndex]), \
            'Log likelihood ' + str(logLik)

    if logLik < oldLogLik and iteration > 10:
        didNotConverge = 0
    if np.isnan(np.sum(mean)):
        print 'NaN error at iteration ', str(iteration)
        didNotConverge = 0
    bootlog('finished iteration '+str(iteration))
  save_file(classProb, path_seg+'new_seg_' + file_name)
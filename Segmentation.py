#! /usr/bin/python
import numpy as np
import numpy as N
import os
from scipy.stats import norm
from PIL import Image
from nifti import *
# import nibabel as nib
from pylab import *

# TODO: I/O, 3rd dim, uMRF
path = '/home/smorrell/git/ipmi/MPHGB06_coursework_part'
path_in = path + '1/images/'
path_out_ff = path + '1/out_ff/'

def read_file(filename):
  imagefile = NiftiImage(filename)
  image_array = imagefile.asarray()
  return image_array.astype('float32')

# def array_to_image(array):
#   minimal_value = np.min(array)
#   maximal_value = np.max(array)
#   if minimal_value < 0 or maximal_value > 255:
#     array = 255 * (array - minimal_value) / (maximal_value - minimal_value)
#   array_uint8 = array.astype('uint8')
#   return Image.fromarray(array_uint8, 'L')

def save_file(array, filename):
  # coord_transform = np.eye(4)  # TODO: do we need a transform?
  nim = NiftiImage(array)
  nim.save(filename)
  # imagefile = array_to_image(array)  # TODO: do we need to normalise?
  # imagefile.save(filename)

def uMRF (pik, k):
  """Calcualtes the Markov Random Field for the whole image for one class k of the central pixel
  Args: pik: class probability per pixel for whole image with dims [x, y, z, numclass]
  For each pixel (outer 3 lops) we sum the class probs by neighbour & central class into a matrix then * G
  Returns: 
  G energy matrix. penalty for being near pixels. I.e. different = 1, same = 0.  
  Horizontal axis is class j of neighbour. Vertical axis is class k of central pixel."""
  G = np.ones(6, 6) - np.eye(6)
  umrf = np.ndarray([np.size(pik, 0), np.size(pik, 1), np.size(pik, 2)])
  for x in range(0, np.size(pik, 0)):
    for y in range(0, np.size(pik, 1)):
      for z in range(0, np.size(pik, 2)):
        umrfAtPixel = 0
        for j in range(0, numclass):  # for the class of each neighbouring pixel
          umrfj = 0
          if x > 0:
            umrfj += pik[x-1, y, z, j]
          if x+1 < np.size(pik, 0):
            umrfj += pik[x+1, y, j, j]
          if y > 0:
            umrfj += pik[x, y-1, z, j]
          if y+1 < np.size(pik, 1):
            umrfj += pik[x, y+1, z, j]
          if z > 0:
            umrfj += pik[x, y, z-1, j]
          if z+1 < np.size(pik, 2):
            umrfj += pik[x, y, z+1]
          umrfAtPixel += umrfj * G[k, j]
        umrf[x, y, z] = umrfAtPixel
  return umrf

file_name = '1056_F_71.22_AD_60740.nii'
print "Loading image", file_name
imgData = read_file(path_in + file_name)
# Priors
image_prior = read_file(path_out_ff + 'ref_t_flo_new_image_nrr_cpp' + file_name)
image_prior[image_prior == 0] = 10**-7
# GM_Prior = read_file("EM_Lecture/GM_prior.png")
# WM_Prior = read_file("EM_Lecture/WM_prior.png")
# CSF_Prior = read_file("EM_Lecture/CSF_prior.png")
# Other_Prior = read_file("EM_Lecture/NonBrain_prior.png")

didNotConverge = 1
numclass = 4

# Allocate space for the posteriors
classProb = np.ndarray([np.size(imgData, 0), np.size(imgData, 1), numclass])  # r x c x 4
classProbSum = np.ndarray([np.size(imgData, 0), np.size(imgData, 1)])  # r x c

# Allocate space for the priors if using them
classPrior = np.ndarray([np.size(imgData, 0), np.size(imgData, 1), 4])       # r x c x 4
# classPrior = np.ones((np.size(imgData, 0), np.size(imgData,1)))
# classPrior2 = np.ones(np.size(imgData,0))
classPrior[:, :, 0] = GM_Prior/255
classPrior[:, :, 1] = WM_Prior/255
classPrior[:, :, 2] = CSF_Prior/255
classPrior[:, :, 3] = Other_Prior/255

# initialise mean and variances
# mean = np.random.rand(numclass, 1) * 256  # 4x1
mean = np.reshape(np.mean(classPrior, (0, 1)), [4, 1]) * 256 + 100  # why does +100 increase convergence?
# var = (np.random.rand(numclass, 1) * 10) + 200
var = (np.var(classPrior, (0, 1)) * 10) + 200

logLik = -1000000000
oldLogLik = -1000000000
iteration = 0

# Iterative process
while didNotConverge:
  iteration += 1

  # Expectation
  classProbSum[:, :] = 0;
  for classIndex in range(0, numclass):
    # for MRF, replace classPrior below with pi exp(-beta*Umrf()) / normalising term
    gaussPdf = norm.pdf(imgData - mean[classIndex], scale=np.sqrt(var[classIndex])) * classPrior[:, :, classIndex]
    classProb[:, :, classIndex] = gaussPdf
    classProbSum[:, :] += gaussPdf

  # normalise posterior
  for classIndex in range(0, numclass):
    classProb[:, :, classIndex] = classProb[:, :, classIndex] / classProbSum[:, :]

  # Cost function
  oldLogLik = logLik
  logLik = np.sum(np.log(classProbSum))  # slide 47 equation 2

  # Maximization
  for classIndex in range(0, numclass):
    pik = classProb[:, :, classIndex]
    mean[classIndex] = np.sum(pik*imgData) / np.sum(pik)
    var[classIndex]  = np.sum(pik*(imgData-mean[classIndex])**2) / np.sum(pik)

    print str(classIndex) + " = " + str(mean[classIndex]) + " , " + str(var[classIndex])

  if logLik < oldLogLik:
      didNotConverge = 0
  if np.isnan(np.sum(mean)):
      didNotConverge = 0

print iteration
save_file(classProb[:, :, 0] * 255, "seg0.png")
save_file(classProb[:, :, 1] * 255, "seg1.png")
save_file(classProb[:, :, 2] * 255, "seg2.png")
save_file(classProb[:, :, 3] * 255, "seg3.png")

""" use the transformation from image template to individual images.
Apply these to the labels to get transformed labels
Get the jacobian from the first transform. Extract a subset voxels for each class using index. Average the subsets.
"""
import subprocess
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
path_out_jac = path + '1/jac/'
path_out_labels = path + '1/out_labels/'
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

def by_my_irroyal(command):
  with open('test.log', 'w') as f:
    process = subprocess.Popen(command.split(), stdout=subprocess.PIPE)
    for c in iter(lambda: process.stdout.read(1), ''):
      sys.stdout.write(c)
      f.write(c)
  output, error = process.communicate()

def jacs_per_region():
  file_name = '1056_F_71.22_AD_60740.nii'
  jacobians = read_file(path_out_jac + 'jac' + file_name)
  transformed_labels = read_file(path_out_labels + 'propagated_labels' + file_name)
  
  print 1

if __name__ == '__main__':
  jacs_per_region()
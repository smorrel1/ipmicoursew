from time import time
from nifti import *
import numpy as np

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

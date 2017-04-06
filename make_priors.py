from nifti import *
import numpy as N
from pylab import *
# propagate the provided probabilistic tissue maps to all individual subject scans and use them as priors
# 	-aff <filename>		Filename which contains the output affine transformation. [outputAffine.txt]
# 	-res <filename>		Filename of the resampled image. [outputResult.nii]

# ref_image = '/home/smorrell/Dropbox/PhD/IPMI/ipmicoursew/MPHGB06_coursework_part1/images/1056_F_71.22_AD_60740.nii'
import subprocess
import os
path = '/home/smorrell/git/ipmi/MPHGB06_coursework_part'
path_in = path + '1/images/'
path_ave = path + '2/average/'
path_out_rigid = path + '1/out_rigid/'
path_out_ff = path + '1/out_ff/'
path_out_labels = path + '1/out_labels/'
path_out_priors = path + '1/out_priors/'

def by_my_irroyal(command):
  with open('test.log', 'w') as f:
    process = subprocess.Popen(command.split(), stdout=subprocess.PIPE)
    for c in iter(lambda: process.stdout.read(1), ''):
      sys.stdout.write(c)
      f.write(c)
  output, error = process.communicate()

def linear_registrations():
  counter = 0
  for file_name in os.listdir(path_in):
    if counter > 5:
      break
    command = 'reg_aladin -ref ' + path_in + file_name + \
            ' -flo ' + path_ave + 'average_template.nii ' + \
            '-res ' + path_out_rigid + 'ref_t_flo_p_affine_result' + file_name + \
            ' -aff ' + path_out_rigid + 'ref_t_flo_p_affine_matrix' + file_name[:-4] + '.txt'
    by_my_irroyal(command)

def non_linear_registration():
  for file_name in os.listdir(path_in):
    command = 'reg_f3d -ref ' + path_in + file_name + \
              ' -flo ' + path_ave + 'average_template.nii ' + \
              ' -res ' + path_out_ff + 'ref_t_flo_p_affine_result' + file_name + \
              ' -aff ' + path_out_rigid + 'ref_t_flo_p_affine_matrix' + file_name[:-4] + '.txt' \
              ' -cpp ' + path_out_ff + 'ref_t_flo_new_image_nrr_cpp' + file_name
              # 	-cpp <filename>		Filename of control point grid [outputCPP.nii]
    by_my_irroyal(command)

def resample_priors():
  for file_name in os.listdir(path_in):
    command = 'reg_resample -ref ' + path_in + file_name + ' -flo ' + path_ave + 'average_priors.nii' + \
              ' -res ' + path_out_priors + 'propagated_priors' + file_name + \
              ' -trans ' + path_out_ff + 'ref_t_flo_new_image_nrr_cpp' + file_name + ' -inter 0'
#	-trans Filename of the file containing the transformation parametrisation (from reg_aladin, reg_f3d or reg_transform)
    by_my_irroyal(command)

def show_images():
  file_name = os.listdir(path_in)[0]
  nim = NiftiImage(path_in+file_name)
  print 'first subject', nim.header['dim']
  imshow(nim.data[90], interpolation='nearest')  #, cmap=cm.grey)  # slice through the vertical axis
  show()
  print 'transformed image'
  transf = NiftiImage(path_out_rigid + 'ref_t_flo_p_affine_result' + file_name)
  imshow(nim.data[90], interpolation='nearest')  #, cmap=cm.grey)  # slice through the vertical axis
  show()
  template = NiftiImage(path_ave + 'average_template.nii')
  print 'tissue map average template'
  imshow(template.data[90], interpolation='nearest')  #, cmap=cm.grey)  # slice through the vertical axis
  show()

def make_difference_images():
  pass

if __name__ == '__main__':
  # linear_registrations()
  # show_images()
  # non_linear_registration()
  resample_priors()

# dims: ndim, x, y, z, t, u, v, w axis.  reversed.
# print imgData.filename
# imgData.setFilename()
# noise = N.random.randn(100, 16, 32, 32)
# imgData = NiftiImage(noise)
# nim1 = NiftiImage(ref_image)
# print nim1.header['dim']
# print imgData.header['datatype']  == 64  # nifticlib.NIFTI_TYPE_FLOAT64
# imgData.save('/home/smorrell/Dropbox/PhD/IPMI/ipmicoursew/MPHGB06_coursework_part1/images/noise.nii.gz')
# nim2 = NiftiImage(imgData.data[:10], imgData.header)
# nim2.save('/home/smorrell/Dropbox/PhD/IPMI/ipmicoursew/MPHGB06_coursework_part1/images/part.hdr.gz')
# imshow(imgData.data[::-1, :, 90], interpolation='nearest')  # slice through the z (front to back) axis x = 90

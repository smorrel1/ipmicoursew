import numpy as np

def read_file(_):
  array = np.ones([1, 2])
  array[0, 0] = 0
  return array
""""""
path = '/home/smorrell/git/ipmi/MPHGB06_coursework_part'
path_out_labels = path + '1/out_labels/'
file_name = '1056_F_71.22_AD_60740.nii'
label_array = read_file(path_out_labels + 'propagated_labels' + file_name)
label_list_unique = np.unique(label_array).astype(np.uint32)
print 'label_array', label_array
print 'labels', label_list_unique
ave_jacobian = np.zeros(len(label_list_unique))  # 1-dim array, starting at index 0 mapped to 1st class.
jacobian = np.ones([1, 2])*.9
print 'jacobian', jacobian
for label_j in label_list_unique:
  print 'for label:', label_j
  x = np.average(jacobian[label_array == label_j])
  ave_jacobian[label_j] = x
  print ave_jacobian[label_j]
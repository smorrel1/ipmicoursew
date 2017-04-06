#! /usr/bin/python

import numpy as np
import os
import nibabel as nib
from PIL import Image

#img = nib.read_file('1056_F_71.22_AD_60740.nii')

def read_file(data_path, image_name):
    filename = os.path.join(data_path, image_name)
    image = nib.load(filename)
    image_array = image.get_data()
    #imagefile = Image.open(filename)
    #image_array = np.array(imagefile.getdata(), np.uint8).reshape(imagefile.size[1], imagefile.size[0])
    return image_array.astype('float32')

def array_to_image(array):
    minimal_value = np.min(array)
    maximal_value = np.max(array)
    if minimal_value < 0 or maximal_value > 255:
        array = 255*(array-minimal_value)/(maximal_value-minimal_value)
    array_uint8 = array.astype('uint8')
    return Image.fromarray(array_uint8, 'L')

def save_file(array,filename):
    coordinate_transform = np.eye(4)
    imagefile = nib.Nifti1Image(array, coordinate_transform)
    nib.save(filename,"/Users/laurariggall/Desktop/niftyreg_install/bin/" )
#    imagefile = array_to_image(array)
#    imagefile.save(filename)

def uMRF (pik, k):  #  TODO: add z dimension to arrays and inside j loop.  Why is k passed but not used?
    G = np.ones((4, 4))-np.diag([1,1,1,1])
    umrf = np.ndarray([np.size(pik,0),np.size(pik,1),np.size(pik,2)])
    for x in range(0, np.size(pik,0)):    
        for y in range (0, np.size(pik,1)):
            for z in range (0, np.size(pik,2)):
                    umrfAtIndex = 0
                    for j in range (0, numclass): #j=no. classes
                        # sum neighbouring probabilities
                        umrfj = 0
                        if (y-1)>-1:
                            umrfj += pik[x,y-1,z,j] #left
                        if (y+1)<np.size(pik,1):
                            umrfj += pik[x,y+1,z,j] #right
                        if (x-1)>-1:
                            umrfj += pik[x-1,y,z,j] #below
                        if (x+1)<np.size(pik,0):
                            umrfj += pik[x+1,y,z,j] #above
                          # TODO: add z if statement
                        umrfAtIndex += umrfj * G[k, j]
                    umrf[x, y, z] = umrfAtIndex #umrf over all j

    return umrf
    
#print( "Loading data")
imgData = read_file("/Users/laurariggall/Desktop/niftyreg_install/bin/","1056_F_71.22_AD_60740.nii") #calls read_file function

# Priors
image_Prior = read_file("/Users/laurariggall/Desktop/niftyreg_install/bin/","1056prior.nii")
image_Prior[image_Prior == 0] = 10**-7
#print(image_Prior)   
#GM_prior
#WM_Prior = read_file("WM_prior.png")
#CSF_Prior = read_file("CSF_prior.png")
#Other_Prior = read_file("NonBrain_prior.png")


didNotConverge = 1 # when zero, iterations stop
numclass = 4 # number of tissue classes
#
## Allocate space for the posteriors
classProb=np.ndarray([np.size(imgData,0),np.size(imgData,1), np.size(imgData,2), numclass]) #same size as image in x, y and z, no. of classes.
classProbSum=np.ndarray([np.size(imgData,0),np.size(imgData,1), np.size(imgData,2)]) #same size as image
classProb[classProb == 0] = 10**-20
#print(classProb)
#a[a < 0] = 0
# Allocate space for the priors if using them
#classPrior=np.ndarray([np.size(imgData,0),np.size(imgData,1),np.size(imgData,2),4])
#classPrior[:, :, :, 0] = GM_Prior/255
#classPrior[:, :, :, 1] = WM_Prior/255
#classPrior[:, :, :, 2] = CSF_Prior/255
#classPrior[:, :, :, 3] = Other_Prior/255

##initialise mean and variances
mean = np.zeros(numclass);
var= np.zeros(numclass);
for classIndex in range(0, numclass):
        pik = image_Prior[:,:,:,classIndex]
        mean[classIndex] = np.sum(pik*imgData)/np.sum(pik)
        var[classIndex] = np.sum(pik * ((imgData - mean[classIndex])**2))/np.sum(pik)

# Initial log likelihood
logLik=-1000000000
oldLogLik=-1000000000

# Initial values for uMRF calculation
beta = 0.5 #scaling factor used for bias field correction
MRF = np.ndarray([np.size(imgData,0),np.size(imgData,1),np.size(imgData,2), numclass])
MRF[:,:,:,:] = 1

## Iterative process
iteration=0
while didNotConverge:
    iteration=iteration+1
    
    # Expectation
    classProbSum[:, :] = 0;
    for classIndex in range(0, numclass):
        gaussPdf = 1/np.sqrt(2*np.pi*var[classIndex]) * np.exp(-(imgData - mean[classIndex])**2/(2*var[classIndex])) 
        #np.sqrt is an element-wise opperator
        classProb[:, :, :, classIndex] = gaussPdf * image_Prior[:,:,:,classIndex] * MRF[:,:,:,classIndex]
        classProbSum[:, :,:] = classProbSum[:, :,:] + classProb[:, :, :, classIndex]
    classProbSum[classProbSum == 0] = 10**-7 
    classProbSum[classProbSum < 0] = 10**-7 
    # normalise posterior ( = pik, not stores as a new variable in order to save memory)
    for classIndex in range(0, numclass):
        classProb[:, :, :, classIndex] = classProb[:,:,:,classIndex]/classProbSum[:,:,:] #starts dividing by zero here which gives runtime warning
    classProb[classProb == 0] = 10**-7
    # Cost function
    oldLogLik = logLik
    logLik = np.sum(np.log(classProbSum)) # sum over all elements

    # Maximization
    for classIndex in range(0, numclass):
        pik = classProb[:,:,:,classIndex]
        mean[classIndex] = np.sum(pik*imgData)/np.sum(pik)
        var[classIndex] = np.sum(pik * ((imgData - mean[classIndex])**2))/np.sum(pik)
        MRF[:,:,:,classIndex] = np.exp(-beta * uMRF(classProb, classIndex))

        print(str(classIndex)+" = "+str(mean[classIndex])+" , "+str(var[classIndex]))


    if iteration>1:#change this back to greater than ten
        didNotConverge=0

for classIndex in range(0, numclass):
    save_file(classProb[ : ,: ,:,classIndex] * 255, "seg"+str(classIndex)+".nii")

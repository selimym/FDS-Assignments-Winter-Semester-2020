import numpy as np
from numpy import histogram as hist
from scipy.signal import convolve


#Add the Filtering folder, to import the gauss_module.py file, where gaussderiv is defined (needed for dxdy_hist)
import sys, os, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
filteringpath = os.path.join(parentdir, 'Filtering')
sys.path.insert(0,filteringpath)
import gauss_module
import math
import matplotlib.pyplot as plt


#  compute histogram of image intensities, histogram should be normalized so that sum of all values equals 1
#  assume that image intensity varies between 0 and 255
#
#  img_gray - input image in grayscale format
#  num_bins - number of bins in the histogram
def normalized_hist(img_gray, num_bins):
    assert len(img_gray.shape) == 2, 'image dimension mismatch'
    assert img_gray.dtype == 'float', 'incorrect image type'

    hists = [0] * (num_bins )
    range_bin = 255/num_bins

    for x in img_gray:
        for pix in x:
            hists[int(pix / range_bin)] += 1
    bins = []
    t = 0.0
    while(t < 255):
        bins.append(t)#
        t += range_bin
    bins.append(255.0)
    bins = np.array(bins)
    s = sum(hists)
    for j in range(len(hists)):
        hists[j] /= s
    hists = np.array(hists)

    return hists, bins


#  Compute the *joint* histogram for each color channel in the image
#  The histogram should be normalized so that sum of all values equals 1
#  Assume that values in each channel vary between 0 and 255
#
#  img_color - input color image
#  num_bins - number of bins used to discretize each channel, total number of bins in the histogram should be num_bins^3
#
#  E.g. hists[0,9,5] contains the number of image_color pixels such that:
#       - their R values fall in bin 0
#       - their G values fall in bin 9
#       - their B values fall in bin 5
def rgb_hist(img_color_double, num_bins):
    assert len(img_color_double.shape) == 3, 'image dimension mismatch'
    assert img_color_double.dtype == 'float', 'incorrect image type'

    #... (your code here)


    #Define a 3D histogram  with "num_bins^3" number of entries
    hists = np.zeros((num_bins, num_bins, num_bins))
    range_bin = 255 / num_bins
    # Increment the histogram bin which corresponds to the R,G,B value of the pixel i
    for x in img_color_double:
        for pix in x:
            hists[int(pix[0]/range_bin),int(pix[1]/range_bin),int(pix[2]/range_bin)]+=1
            #print(pix) #triple






    #Normalize the histogram such that its integral (sum) is equal 1
    #print(,hists.shape)+
    s=np.sum(hists)
    for x in range(len(hists)):
        for y in range(len(hists[x])):
            for z in range(len(hists[x][y])):
                hists[x][y][z] /= s
    #... (your code here)

    #Return the histogram as a 1D vector
    hists = hists.reshape(hists.size)#16384.0

    return hists



#  Compute the *joint* histogram for the R and G color channels in the image
#  The histogram should be normalized so that sum of all values equals 1
#  Assume that values in each channel vary between 0 and 255
#
#  img_color - input color image
#  num_bins - number of bins used to discretize each channel, total number of bins in the histogram should be num_bins^2
#
#  E.g. hists[0,9] contains the number of image_color pixels such that:
#       - their R values fall in bin 0
#       - their G values fall in bin 9
def rg_hist(img_color_double, num_bins):

    assert len(img_color_double.shape) == 3, 'image dimension mismatch'
    assert img_color_double.dtype == 'float', 'incorrect image type'
    hists = np.zeros((num_bins, num_bins))
    range_bin = 255 / num_bins
    #print(num_bins)
    for x in img_color_double:
        for pix in x:
            hists[int(pix[0] / range_bin), int(pix[1] / range_bin)] += 1
            # print(pix) #triple

    #Define a 2D histogram  with "num_bins^2" number of entries
    #hists = np.zeros((num_bins, num_bins))
    s = np.sum(hists)
    for x in range(len(hists)):
        for y in range(len(hists[x])):

                hists[x][y] /= s

    #Return the histogram as a 1D vector
    hists = hists.reshape(hists.size)

    return hists




#  Compute the *joint* histogram of Gaussian partial derivatives of the image in x and y direction
#  Set sigma to 3.0 and cap the range of derivative values is in the range [-6, 6]
#  The histogram should be normalized so that sum of all values equals 1
#
#  img_gray - input gray value image
#  num_bins - number of bins used to discretize each dimension, total number of bins in the histogram should be num_bins^2
#
#  Note: you may use the function gaussderiv from the Filtering exercise (gauss_module.py)
def dxdy_hist(img_gray, num_bins):
    assert len(img_gray.shape) == 2, 'image dimension mismatch'
    assert img_gray.dtype == 'float', 'incorrect image type'


    #... (your code here)


    #Define a 2D histogram  with "num_bins^2" number of entries
    #hists = np.zeros(num_bins)
    hists = np.zeros((num_bins, num_bins))

    img_smoothx,img_smoothy = gauss_module.gaussderiv(img_gray,3)


    range_bin = 13 / num_bins   #the range of coloro for a pixel is [-6,6]

    for x in range(len(img_smoothx)):
        for y in range (len(img_smoothx[x])):
            if img_smoothx[x][y] > 6:
                img_smoothx[x][y] = 6
            if img_smoothx[x][y] < -6:
                img_smoothx[x][y] = -6
            if img_smoothy[x][y] > 6:
                img_smoothy[x][y] = 6
            if img_smoothy[x][y] < -6:
                img_smoothy[x][y] = -6
            #we add 6 becouse a negative index will write backwords in the array
            #in this way 0 is the middle of the histograma separating negative and positive values
            a = int((img_smoothx[x][y] + 6) / range_bin)
            b = int((img_smoothy[x][y] + 6) / range_bin)
            hists[a, b] += 1

    # Define a 2D histogram  with "num_bins^2" number of entries
    # hists = np.zeros((num_bins, num_bins))
    s = np.sum(hists)
    for x in range(len(hists)):
        for y in range(len(hists[x])):
            hists[x][y] /= s



    #Return the histogram as a 1D vector
    hists = hists.reshape(hists.size)
    return hists

def is_grayvalue_hist(hist_name):
  if hist_name == 'grayvalue' or hist_name == 'dxdy':
    return True
  elif hist_name == 'rgb' or hist_name == 'rg':
    return False
  else:
    assert False, 'unknown histogram type'


def get_hist_by_name(img, num_bins_gray, hist_name):
  if hist_name == 'grayvalue':
    return normalized_hist(img, num_bins_gray)
  elif hist_name == 'rgb':
    return rgb_hist(img, num_bins_gray)
  elif hist_name == 'rg':
    return rg_hist(img, num_bins_gray)
  elif hist_name == 'dxdy':
    return dxdy_hist(img, num_bins_gray)
  else:
    assert False, 'unknown distance: %s'%hist_name


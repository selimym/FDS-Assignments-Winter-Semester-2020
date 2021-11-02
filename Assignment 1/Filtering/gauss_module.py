# import packages: numpy, math (you might need pi for gaussian functions)
import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.signal import convolve2d as conv2
from scipy.signal import convolve


"""
Gaussian function taking as argument the standard deviation sigma
The filter should be defined for all integer values x in the range [-3sigma,3sigma]
The function should return the Gaussian values Gx computed at the indexes x
"""
def gauss(sigma):
    sigma = int(sigma)

    x = []
    Gx = []
    for i in range(-3 * sigma, 3 * sigma + 1):
        x.append(i)
        Gx.append((1 / (math.sqrt(2 * math.pi) * sigma)) * math.exp(pow(i,2) / (2 * pow(sigma,2)) * -1))
    Gx = np.array(Gx)
    return Gx, x



"""
Implement a 2D Gaussian filter, leveraging the previous gauss.
Implement the filter from scratch or leverage the convolve2D method (scipy.signal)
Leverage the separability of Gaussian filtering
Input: image, sigma (standard deviation)
Output: smoothed image
"""
def gaussianfilter(img, sigma):
    
    Gx,_ = gauss(sigma)

    smooth_img = np.empty_like(img) #deep copy to not modify img
    np.copyto(smooth_img, img)
    x,y =img.shape

    for row in range(x):
       smooth_img[row] = convolve(smooth_img[row], Gx, mode="same")  #orizontal filter
    smooth_img = np.transpose(smooth_img)
    for col in range(y):
        smooth_img[col] = convolve(smooth_img[col], Gx, mode="same")  #vertical filter
    smooth_img = np.transpose(smooth_img)

    return smooth_img



"""
Gaussian derivative function taking as argument the standard deviation sigma
The filter should be defined for all integer values x in the range [-3sigma,3sigma]
The function should return the Gaussian derivative values Dx computed at the indexes x
"""
def gaussdx(sigma):
    sigma = int(sigma)

    x = []
    Dx = []
    Gx,_ = gauss(sigma)
    #Dx = convolve(Gx, [1,0,-1], mode="same")
    for i in range(-3 * sigma, 3 * sigma + 1):
        x.append(i)
        Dx.append((-1*i / (math.sqrt(2*math.pi) * pow(sigma,3)) * math.exp(-1*pow(i,2) / (2*pow(sigma,2)))))
    Dx = np.array(Dx)
    return Dx, x



def gaussderiv(img, sigma):

    #...
    Dx,_ = gaussdx(sigma)
    #Dx = np.array(Dx)


    x,y =img.shape
    imgDx=np.empty_like(img)
    #imgDy = np.empty_like(img)
    np.copyto(imgDx,img)
    #np.copyto(imgDy, img)
    for row in range(x):
       imgDx[row] = convolve(imgDx[row],Dx, mode="same")  #orizontal filter
    imgDy = np.empty_like(img)
    np.copyto(imgDy, img)
    imgDy=np.transpose(imgDy)
    for col in range(y):

        imgDy[col] = convolve(imgDy[col], Dx, mode="same")  #vertical filter

    imgDy=np.transpose(imgDy)


    return imgDx, imgDy


import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import histogram_module
import dist_module

def rgb2gray(rgb):

    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

    return gray



# model_images - list of file names of model images
# query_images - list of file names of query images
#
# dist_type - string which specifies distance type:  'chi2', 'l2', 'intersect'
# hist_type - string which specifies histogram type:  'grayvalue', 'dxdy', 'rgb', 'rg'
#
# note: use functions 'get_dist_by_name', 'get_hist_by_name' and 'is_grayvalue_hist' to obtain 
#       handles to distance and histogram functions, and to find out whether histogram function 
#       expects grayvalue or color image

def find_best_match(model_images, query_images, dist_type, hist_type, num_bins):

    hist_isgray = histogram_module.is_grayvalue_hist(hist_type)
    model_hists = compute_histograms(model_images, hist_type, hist_isgray, num_bins)

    #img5 = [model_hists[x] for x in [x for x in range(len(model_images))  if model_images[x]=="obj5__0.png"]]
   # print("CONFRONTA CON ------------- 5 0",[model_hists[x] for x in [x for x in range(len(model_images))  if model_images[x]=="model/obj5__0.png"]])
   # print("CONFRONTA CON ------------- 13 0",[model_hists[x] for x in [x for x in range(len(model_images))  if model_images[x]=="model/obj13__0.png"]])
    query_hists = compute_histograms(query_images, hist_type, hist_isgray, num_bins)
    
    D = np.zeros((len(query_images),len(model_images)))

    flag = False;
    best_match = np.zeros(len(query_images))

    for i in range(len(query_images)):  # for each query img


        d_intersect=-1 #fix the min dist value, for intersect the maximum is the best match
        d = 2  # fix the max dist value, for l2 and chi2 the minimum is the best match
        for j in range(len(model_images)):  # for each model img
            actual_d=dist_module.get_dist_by_name(query_hists[i], model_hists[j], dist_type)

            if dist_type == 'intersect':
                if actual_d > d_intersect:  # if this is a better match
                    best_match[i] = j  # (for now) this is the best match
                    d_intersect = actual_d  # update the (new) better distance

            else:
                if actual_d < d:  # if this is a better match
                    best_match[i] = j  # (for now) this is the best match
                    d = actual_d  # update the (new) better distance
            D[i][j] =dist_module.get_dist_by_name(query_hists[i], model_hists[j], dist_type)   # fill the matr
        flag = False;

    return best_match, D



def compute_histograms(image_list, hist_type, hist_isgray, num_bins):
    
    image_hist = []
    for img in image_list:
        img_color = np.array(Image.open(img))

        if hist_isgray:
            img_gray = rgb2gray(img_color.astype('double'))
            image_hist.append(histogram_module.get_hist_by_name(img_gray, num_bins, hist_type))

        else:

            image_hist.append(histogram_module.get_hist_by_name(img_color.astype('double'), num_bins, hist_type))


    return image_hist



# For each image file from 'query_images' find and visualize the 5 nearest images from 'model_image'.
#
# Note: use the previously implemented function 'find_best_match'
# Note: use subplot command to show all the images in the same Python figure, one row per query image

def show_neighbors(model_images, query_images, dist_type, hist_type, num_bins):
    bm, D = find_best_match(model_images, query_images, dist_type, hist_type, num_bins)
    x,y = D.shape
    new_d = np.empty_like(D)
    np.copyto(new_d,D) #not touching D!


    for i in range(len(query_images)):
        best_five = []
        img = np.array(Image.open(query_images[i]))
        if dist_type=='intersect':
            for j in range(5):
                index=np.argmax(new_d[i])
                best_five.append(index)
                new_d[i][index] = -1
        else:
            for j in range(5):
                index = np.argmin(new_d[i])
                best_five.append(index)
                new_d[i][index] = 2

        plt.figure()
        plt.subplot(1, 6, 1);
        plt.imshow(np.array(img));
        plt.subplot(1, 6, 2);

        plt.imshow(np.array(Image.open(model_images[best_five[0]])));
        plt.subplot(1, 6, 3);
        plt.imshow(np.array(Image.open(model_images[best_five[1]])));
        plt.subplot(1, 6, 4);
        plt.imshow(np.array(Image.open(model_images[best_five[2]])));
        plt.subplot(1, 6, 5);
        plt.imshow(np.array(Image.open(model_images[best_five[3]])));
        plt.subplot(1, 6, 6);
        plt.imshow(np.array(Image.open(model_images[best_five[4]])));
        plt.show()

            #what should we do if type_dist isn't intersect?
            #print the 5 smaller value of D for each query img



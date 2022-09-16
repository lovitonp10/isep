# System imports
import matplotlib.pyplot as plt
import numpy as np
import datetime
from time import process_time

#Extra library imports
from PIL import Image
from scipy import optimize
import scipy.ndimage as ndi
import pandas as pd
import seaborn as sns
from scipy import ndimage
    
from skimage import filters
from skimage.filters import threshold_otsu 

def iou_circle_analysis(impath):
    from skimage import filters
    from skimage.filters import threshold_otsu 
    im = Image.open(impath)
    imarray = np.array(im) 
    imarray.shape 
    plt.imshow(imarray)

    mask1_arr = (imarray >= 0.05 * np.max(imarray)).astype(int)

    # Sobel Kernel
    ed_sobel = filters.sobel(mask1_arr)

    threshold_value = threshold_otsu(ed_sobel)  

    threshold_ed_sobel = ed_sobel > threshold_value 
    
    '''plt.imshow(imarray)
    plt.contour(threshold_ed_sobel, colors = 'r')
    plt.plot(y0,x0, marker="o", color="r")
    plt.title('Microscopic image and the tumor contours')
    plt.show()'''
    [x0, y0] = np.where(mask1_arr == 1)
    x0 = np.mean(x0)
    y0 = np.mean(y0)
    [x, y] = np.where(threshold_ed_sobel == 1)
    # Initialize the radius
    r0 = np.mean(np.sqrt((x-x0)**2+(y-y0)**2))

    # Create the binary image of the (filled) circle of center (x0, y0) and radius r0
    sx, sy = threshold_ed_sobel.shape
    y_arr, x_arr = np.meshgrid(range(sy), range(sx))

    circle_arr = (((x_arr - x0)**2+(y_arr-y0)**2)<=r0**2).astype(int)
    
    def predict(P, alpha, beta, gamma):

        """
        The function to fit 


        Parameters
        ----------
        P: 2xn array
            the array containing the x (first line) and y (second line) values 
        alpha: float
            the alpha parameter
        beta: float
            the beta parameter
        gamma: float
            the gamma parameter

        Returns
        -------
        pred_p: vector
            the values of the fitted input data (the z values)

        Command:
        pred_p = predict(P, alpha, beta, gamma)

        """

        # Divide the input vector into x and y
        x, y = P[0, :], P[1, :]

        # Vector of predictions according to the model 
        pred_p = alpha*x+beta*y+gamma

        return pred_p
    
    # Concatenate x and y
    P = np.vstack((x, y))

    # Compute the vector of the z_i values
    x = P[0, :]
    y = P[1, :]

    Z = x**2+y**2

    # Initialize the vector of variable values
    p0 = [2*x0, 2*y0, r0**2-x0**2-y0**2]
    alpha = p0[0]
    beta = p0[1]
    gamma = p0[2]
    # Display the initial alpha, beta and gamma values
    #print("The initial alpha is", p0[0])
    #print("The initial beta is", p0[1])
    #print("The initial gamma is", p0[2])

    # Display the initial loss value
    loss_val = 0.5*np.mean((Z-predict(P,*p0))**2)
    print("The initial loss value is", loss_val)

    # Fit the model using the Levenberg-Marquard method
    lev_marq = optimize.curve_fit(predict,P,Z,p0=p0)

    # Display the optimized alpha, beta and gamma values
    alpha_final = lev_marq[0][0]
    beta_final = lev_marq[0][1]
    gamma_final = lev_marq[0][2]

    #print("The optimized alpha is",alpha_final)
    #print("The optimized beta is", beta_final)
    #print("The optimized gamma is", gamma_final)

    # Compute and display the final loss value
    Z_pred = predict(P,alpha_final,beta_final,gamma_final)
    loss_val_final = 0.5*np.mean((Z-Z_pred)**2)
    #print("The final loss value is",loss_val_final)

       # Retrieve the parameter values
    xc = alpha_final/2
    yc = beta_final/2
    r = np.sqrt(gamma_final+xc**2+yc**2)
    #print(x0, y0, r0)
    #print(xc, yc, r)
    
    
    # Compute the binary image of the optimized fitted circle
    circle_arr_optimized = (((x_arr - xc)**2+(y_arr-yc)**2)<=r**2).astype(int)
    
    iou = np.sum(circle_arr_optimized&mask1_arr)/np.sum(circle_arr_optimized|mask1_arr)

    return iou
    
    
    
    
    
    
    

from skimage.filters import threshold_otsu 
from skimage.feature import canny 

def all_contour_length_analysis(impath):
    im = Image.open(impath)
    imarray = np.array(im) 
    imarray.shape 
    plt.imshow(imarray)
    
    # Canny Kernel
    can = canny(imarray)
 
    threshold_value = threshold_otsu(can)  

    threshold_can = can > threshold_value  
    plt.imshow(threshold_can)
    
    return np.sum(threshold_can)




from skimage import filters
from skimage.filters import threshold_otsu 

def contour_length_analysis(impath):
    im = Image.open(impath)
    imarray = np.array(im) 
    imarray.shape 
    plt.imshow(imarray)
    
    mask1_arr = (imarray >= 0.05 * np.max(imarray)).astype(int)

    # Sobel Kernel
    ed_sobel = filters.sobel(mask1_arr)
 
    threshold_value = threshold_otsu(ed_sobel)  

    threshold_ed_sobel = ed_sobel > threshold_value  
    plt.imshow(threshold_ed_sobel)
    
    return np.sum(threshold_ed_sobel)




def symmetry_analysis(impath):
    im = Image.open(impath)
    imarray = np.array(im) 
    imarray.shape 
    plt.imshow(imarray)
    
    mask1_arr = (imarray >= 0.05 * np.max(imarray)).astype(int)
    
    # Display the obtained mask
    #fig, ax = plt.subplots(figsize=(10, 10))
    #ax.imshow(mask1_arr, cmap='gray')
    #ax.axis('off')
    #plt.title('A first mask')
    #plt.show()
    
    
    
    sx1, sy1 = imarray.shape
    # Compute the connected components (i.e regions) of the complementary image
    ccs_arr, nb_ccs = ndi.label(1 - mask1_arr)

    # List of ccs indexes 
    cc_ids = list(range(1, nb_ccs + 1))

    # Compute the array of the border of the image 
    border_arr = np.zeros((sx1, sy1)).astype(int)
    border_arr[0, :] = 1
    border_arr[sx1-1, :] = 1
    border_arr[:, 0] = 1
    border_arr[:, sy1-1] = 1

    # Get the indexes of the regions touching the border
    border_cc_ids = np.unique(ccs_arr[border_arr == 1])
    border_cc_ids = list(border_cc_ids[border_cc_ids > 0])

    #Remove the corresponding indexes
    cc_ids = list(set(cc_ids) - set(border_cc_ids))

    #Fill each detected holes    
    for cc_id in cc_ids:
        mask1_arr[ccs_arr == cc_id] = 1 
        
    #Display the filled mask
    #fig, ax = plt.subplots(figsize=(10, 10))
    #ax.imshow(mask1_arr, cmap='gray')
    #ax.axis('off')
    #plt.title('The filled mask')
    #plt.show()
    
    # Coordinates of the pixels belonging to the mask
    [x1, y1] = np.where(mask1_arr == 1)
    print('The mask is composed of {0} pixels'.format(len(x1)))
    
    # Bounding box of the butterfly mask
    bb1_arr = np.zeros((sx1, sy1)).astype(int)
    bb1_arr[np.min(x1):np.max(x1)+1, np.min(y1):np.max(y1)+1] = 1
    
    #Compute and display the percentage of butterfly pixels included in the bounding box
    percentage_included = (np.sum(mask1_arr[bb1_arr==1]))/len(x1)*100
    print('The percentage of pixels included in the bounding box is {0:.2f}%'.format(percentage_included))
    
    #Compute and display the percentage of the bounding box that is occupied by the butterfly mask
    bb1_pixels = np.sum(bb1_arr)
    percentage = (len(x1)/bb1_pixels)*100
    print('The percentage of the bounding box that is occupied by the mask is {0:.2f}%'.format(percentage))
    
    # Coordinates of the center of the bounding box
    x1_bb_center = int(np.round(0.5 * (np.max(x1) - np.min(x1)) + np.min(x1)))
    y1_bb_center = int(np.round(0.5 * (np.max(y1) - np.min(y1)) + np.min(y1)))

    # Vertical and horizontal lines going through the center of the bb
    line1_arr = np.zeros((sx1, sy1)).astype(int)
    line1_arr[:, y1_bb_center] = 1

    line2_arr = np.zeros((sx1, sy1)).astype(int)
    line2_arr[x1_bb_center, :] = 1
    
    #Display the transformed image
    #fig, ax = plt.subplots(figsize=(10, 10))
    #ax.imshow(imarray)
    #ax.contour(line1_arr, [0.5], colors = 'c', linewidths = [2, 2], linestyles = 'dotted')
    #ax.contour(line2_arr, [0.5], colors = 'b', linewidths = [2, 2], linestyles = 'dotted')
    #ax.contour(bb1_arr, [0.5], colors = 'r', linewidths = [2, 2])
    #plt.title('Within a box')
    #plt.show()
    
    
    
    #Restrict the input images to the butterfly bounding box
    im1_arr = imarray[np.min(x1):np.max(x1)+1, np.min(y1):np.max(y1)+1]
    mask1_arr = mask1_arr[np.min(x1):np.max(x1)+1, np.min(y1):np.max(y1)+1]
    sx1, sy1 = mask1_arr.shape
    print('Cropped image of dimension {0} x {1} pixels'.format(sx1, sy1))

    # Coordinates of the center of the image
    x1_bb_center = int(np.round(sx1 / 2))
    y1_bb_center = int(np.round(sy1 / 2))

    # Vertical and horizontal lines going through the center of the cropped image
    line1_arr = np.zeros((sx1, sy1)).astype(int)
    line1_arr[:, y1_bb_center] = 1

    line2_arr = np.zeros((sx1, sy1)).astype(int)
    line2_arr[x1_bb_center, :] = 1
    
    
    def symmetry_image_v0(p, arr, background_val):

        """ Symmetric image with respect to a line of equation ax + by + c = 0

        Parameters
        ----------
        p: vector of floats
            vector containing the line of symmetry with parameters a, b, c
        arr: 2D array
            the input (binary or grey level) image 
        plane
        background_val : float/int
            the background intensity value

        Returns
        -------
        sym_arr: 2D array
            the output symmetric image
        """

        #Get back the line parameters
        a, b, c = p

        # Image size and type
        s1, s2 = arr.shape
        arr_type = type(arr[0, 0])

        # Coordinates of the region of interest
        [x, y] = np.where(arr > 0)

        # Coordinates (xr, yr) of the symmetric point of (x, y) with respect to
        #  a vertical line in the image system 
        if a == 0 and b != 0:              
            yr = (-c/b)+(-c/b)-y
            xr = x
        # Coordinates (xr, yr) of the symmetric point of (x, y) with respect to  
        # an horizontal line in the image system 
        elif a != 0 and b == 0:        
            xr = (-c/a)+(-c/a)-x
            yr = y

        # Coordinates (xr, yr) of the symmetric point of (x, y) with respect to 
        # any different line     
        else:
            xr = x - (2*a*(a*x + b*y + c))/(a**2+b**2)
            yr = y - (2*b*(a*x + b*y + c))/(a**2+b**2)

        #cast to integer and only keep coordinates within the image ! 
        xr = (np.round(xr)).astype(int)
        yr = (np.round(yr)).astype(int)

        x = x[xr < s1]
        y = y[xr < s1]
        yr = yr[xr < s1]
        xr = xr[xr < s1]

        x = x[yr < s2]
        y = y[yr < s2]    
        xr = xr[yr < s2]
        yr = yr[yr < s2]

        x = x[xr >= 0]
        y = y[xr >= 0]  
        yr = yr[xr >= 0]
        xr = xr[xr >= 0]

        x = x[yr >= 0]
        y = y[yr >= 0] 
        xr = xr[yr >= 0]
        yr = yr[yr >= 0]

        # Take into account the background value
        if background_val == 0:
            sym_arr = np.zeros((s1, s2), dtype=arr_type)
        else:
            sym_arr = background_val * np.ones((s1, s2), dtype=arr_type)

        # Compute the symmetric array
        for i in range(len(xr)):
            sym_arr[xr[i], yr[i]] = arr[x[i], y[i]]

        return sym_arr
    
    # Symmetric images with respect to a horizontal, vertical, or any different line
    input_arr = np.copy(mask1_arr)
    back_val = np.min(input_arr)
    s1, s2 = input_arr.shape
    y_arr, x_arr = np.meshgrid(range(s2), range(s1)) 

    init_ps = [[0, 1, -y1_bb_center], 
               [1, 0, -x1_bb_center],
               [(sy1 - 1) / (sx1 - 1), 1 , -sy1 - 1.5]]

    for init_p in init_ps:
        #Symmetric image with given parameters
        sym_arr = symmetry_image_v0(init_p, input_arr, back_val)

        #Symmetric line array
        a, b, c = init_p
        line_arr = (np.abs(a * x_arr + b * y_arr + c) <= 1).astype(int)
        
        
        
    def symmetry_image(p, arr, background_val):

        """ Symmetry image by a line of equation ax + by + c = 0

        Parameters
        ----------
        p: vector
            vector containing the symmetry line parameters a, b, c
        arr: 2D array
            the input (binary or grey level) image (non negative intensities)
        plane
        background_val : float/int
            the background intensity value

        Returns
        -------
        sym_arr: 2D array
            the output symmetry image
        """

        a, b, c = p

        # Image size
        s1, s2 = arr.shape
        arr_type = type(arr[0, 0])

        # Coordinates of the whole image
        y_arr, x_arr = np.meshgrid(range(s2), range(s1))
        xr = x_arr.flatten()
        yr = y_arr.flatten()

        # Coordinates (x, y) of the input point as a function of its symmetric 
        # point (xr, yr) with respect to a vertical line in the image system 
        if a == 0 and b != 0:     
            y = (-c/b)+(-c/b)-yr
            x = xr

        # Coordinates (x, y) of the input point as a function of its symmetric 
        # point (xr, yr) with respect to an horizontal line in the image system    
        elif a != 0 and b == 0:       
            x = (-c/a)+(-c/a)-xr
            y = yr       
        else:
        # Coordinates (x, y) of the input point as a function of its symmetric 
        # point (xr, yr) with respect to any different line
            x = xr - (2*a*(a*xr + b*yr + c))/(a**2+b**2)
            y = yr - (2*b*(a*xr + b*yr + c))/(a**2+b**2)

        #cast to integer and only keep coordinates within the image ! 
        x = (np.round(x)).astype(int)
        y = (np.round(y)).astype(int)

        xr = xr[x < s1]
        yr = yr[x< s1]
        y = y[x < s1]
        x = x[x < s1]

        xr = xr[y < s2]
        yr = yr[y < s2]    
        x= x[y < s2]
        y= y[y < s2]

        xr = xr[x >= 0]
        yr = yr[x >= 0]  
        y = y[x >= 0]
        x = x[x >= 0]

        xr = xr[y >= 0]
        yr = yr[y >= 0] 
        x = x[y >= 0]
        y = y[y >= 0]

        # Consider the background
        if background_val == 0:
            sym_arr = np.zeros((s1, s2), dtype=arr_type)
        else:
            sym_arr = background_val * np.ones((s1, s2), dtype=arr_type)

        # Create the symmetry array
        for i in range(len(xr)):
            sym_arr[xr[i], yr[i]] = arr[x[i], y[i]]

        return sym_arr
    
    #Check your improved algorithm
    for init_p in init_ps:
        #Symmetric image with given parameters
        sym_arr = symmetry_image(init_p, input_arr, back_val)

        #Symmetric line array
        a, b, c = init_p
        line_arr = (np.abs(a * x_arr + b * y_arr + c) <= 1).astype(int)

        
    #Initialize the vector of loss values
    global losses

    # Define the loss function
    def sym_loss(p, arr=input_arr, mask_arr=mask1_arr, background_val=back_val):   

        """
        Parameters glim1_arr
        ----------
        p: 3-uplet
            vector containing the symmetry line parameters a, b, c
        arr: 2D array
            the input (binary or grey level) image 
        plane
        mask_arr: 2D array
            the input binary mask
        plane
        background_val : float/int
            the background intensity value

        Returns
        -------
        loss: float
            the loss value
        """

        # Image size
        s1, s2 = arr.shape   

        # Symmetric image with respect to the input line
        sym_arr = symmetry_image(p, arr, background_val)

        # Compute the current loss value
        loss = np.mean((arr[mask_arr == 1]-sym_arr[mask_arr == 1])**2)

        #Update the vector of loss values
        losses.append(loss)

        return loss
    
    # Initialize the list containing the loss evolution
    losses = []
    for init_p in init_ps:
        sym_loss(init_p, input_arr, mask1_arr,  back_val)
    print("The best initial loss is :", min(losses))

    sym_arr = symmetry_image(init_p, input_arr, back_val)

    for init_p in init_ps:
        if min(losses)==sym_loss(init_p, input_arr, mask1_arr,  back_val):
            init_p1=init_p
    print("The best associated parameters are : \n a=", init_p1[0],"\n b=", init_p1[1], "\n c=", init_p1[2])

    return percentage, init_p1[0], init_p1[1], init_p1[2]

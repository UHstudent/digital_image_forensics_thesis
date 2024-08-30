#this code is based on the explanation for re-sampling by Haney Farid in his book Photo Forensics (MIT Press (2016))
""" 
    This file was used for development and early testing of the resampling algorithm, as described in 
    "Exposing Digital Forgeries by Detecting Traces of Re-sampling" by Alin C. Popescu and Hany Farid
    &
    "Photo Forensics" (MIT Press 2016) by Hany Farid

    What it does:
    The code reads an image inside the execution map, called "example.jpg"
    The code then calculates a probability map and prompts the user to input point for fourier transform squares.
    The code then displays the probability maps allongside the fourier maps.

    Note: the steps for constructing the fourier maps of the book and paper have both been included, to switch between them, you 
    will need to manually comment and uncomment steps in this script. Sherloq featuers a more user friendly environment.

    How to use:
    install dependencies: pip install -r requirements_resammpling.txt
    run script: python resammpling.py
 """

import numpy as np
import cv2
import random
import matplotlib.pyplot as plt

# @ is matrix multiplication operator in python

#help functions:

#Because we are evaluating the relationship between a pixel and its m x m neighbors, pixels without these neighbors (pixels on
#the edges of the image) are not included in the estimation. In this code, we estimate with a 3x3 filter, 
#and thus, the edges (1 pixel) are not included. hence, we subtract 2 when defining the shape of the matrices F and f
def build_matrices(I):
    k = 0
    F = np.zeros(((I.shape[0] - 2) * (I.shape[1] - 2) ,8)) #the i^th row of the matrix F corresponds to the pixel neighbors of the i^th element of f
    f = np.zeros((I.shape[0] - 2) * (I.shape[1] - 2)) # the vector f is the image strung out in row-order
    for y in range(1, I.shape[0] - 1):
        for x in range(1, I.shape[1] - 1):
            F[k, 0] = I[y-1, x-1]
            F[k, 1] = I[y-1, x]
            F[k, 2] = I[y-1, x+1]
            F[k, 3] = I[y, x-1]
            # skip I(x; y) corresponding to a(3; 3) which is 0
            F[k, 4] = I[y, x+1]
            F[k, 5] = I[y+1, x-1]
            F[k, 6] = I[y+1, x]
            F[k, 7] = I[y+1, x+1]
            f[k] = I[y, x]
            k = k + 1
    return F, f


def compute_residual(a, I, x, y):
    r = I[y, x] - (a[0]*I[y-1, x-1] + a[1]*I[y-1, x] + a[2]*I[y-1, x+1] 
                + a[3]*I[y, x-1]  + a[4]*I[y, x+1] 
                + a[5]*I[y+1, x-1] + a[6]*I[y+1, x] + a[7]*I[y+1, x+1] )
    return r

#global list to store selected points
selected_points = []

def on_click(event):
    global selected_points
    if event.inaxes:
        x, y = int(event.xdata), int(event.ydata)
        selected_points.append((x, y))
        print(f'Selected point: ({x}, {y})')



def generate_window(shape): #fourier filter recommendation from paper
    #generate the rotationally invariant window W(x, y)
    rows, cols = shape
    W = np.zeros((rows, cols))
    center_x, center_y = rows // 2, cols // 2
    max_radius = min(center_x, center_y)
    
    for i in range(rows):
        for j in range(cols):
            r = np.sqrt((i - center_x)**2 + (j - center_y)**2) / max_radius * np.sqrt(2)
            if r < 3 / 4:
                W[i, j] = 1
            elif r <= np.sqrt(2):
                W[i, j] = 0.5 + 0.5 * np.cos(np.pi * (r - 3 / 4) / (np.sqrt(2) - 3 / 4))

    return W

def high_pass_filter(shape): #fourier filter recommendation from paper
    #generate the rotationally invariant highpass filter H
    rows, cols = shape
    H = np.zeros((rows, cols))
    center_x, center_y = rows // 2, cols // 2
    max_radius = np.sqrt(center_x**2 + center_y**2)

    for i in range(rows):
        for j in range(cols):
            r = np.sqrt((i - center_x)**2 + (j - center_y)**2) / max_radius * np.sqrt(2)
            if r <= np.sqrt(2):
                H[i, j] = 0.5 - 0.5 * np.cos(np.pi * r / np.sqrt(2))

    return H

def generate_circular_hanning_window(shape):
        #the idea is to pass a circularly symmetric Hanning window
        #This will reduce the spurious peaks caused by edges (thats why outside the circle: = 0)
        #https://dsp.stackexchange.com/questions/58449/efficient-implementation-of-2-d-circularly-symmetric-low-pass-filter
        rows, cols = shape
        hanning_circular_size = min(rows,cols)
        hanning_window = np.hanning(hanning_circular_size)[:, None] * np.hanning(hanning_circular_size)
        filter_hanning = np.zeros((rows, cols))

        center_x, center_y = rows // 2, cols // 2
        max_radius = min(center_x, center_y)
        
        for i in range(rows):
            for j in range(cols):
                r = np.sqrt((i - center_x)**2 + (j - center_y)**2)
                if r <= max_radius:
                    circular_window[i, j] = hanning_window[i, j]
        return circular_window

#general idea of constructing Fourier map for resampling detection:
#A periodic pattern can be revealed by the presence of highly localized peaks in the magnitude of the Fourier transform.
# in order to better visualize these peaks, the following step are recommended:
# multiply probability image by a circularly symmetric Hanning (or Gaussian) window.
# up-sample the probability image by a factor of two before applying the Fourier transform. 
#High pass-filter the Fourier transform (by setting a small region around the origin to zero), 
#scaled to fill the range [0, 1], and gamma corrected with an exponent of 0,5.
def calculate_fourier_map(process_prob_map):
    #a fourier map is better to examine as a square format, than we can more easily apply circular filters to this map and observe for symmetry
    #hence, we take only the shortest distance selected by the user to turn the area into a square
    x,y = process_prob_map.shape
    #take only the quare portion from the middle of user selected area
    size = min(x, y)
    half_size = size // 2
    center_x, center_y = x // 2, y // 2
    square_fourier_map = process_prob_map[center_x - half_size:center_x + half_size,
                                          center_y - half_size:center_y + half_size]

    print(square_fourier_map.shape)

    #Apply Hanning window
    #the idea is to pass a circularly symmetric Hanning window
    #https://dsp.stackexchange.com/questions/58449/efficient-implementation-of-2-d-circularly-symmetric-low-pass-filter
    hanning_window = np.hanning(square_fourier_map.shape[0])[:, None] * np.hanning(square_fourier_map.shape[1])
    I_hanning = square_fourier_map * hanning_window

    #Upsample the image by a factor of two
    #upsample by 2 is a built in function by pyrUp
    I_upsampled = cv2.pyrUp(I_hanning)

    """ #paper ratationally invariant window
    shape = square_fourier_map.shape
    W = generate_window(shape)

    rotational_invariant_window = square_fourier_map * W """

    """ #Upsample the image by a factor of two
    #upsample by 2 is a built in function by pyrUp
    I_upsampled = cv2.pyrUp(rotational_invariant_window) """

    #Compute Fourier transform
    dft = np.fft.fft2(I_upsampled)
    fourier = np.fft.fftshift(dft)

    #Apply high-pass filter
    #create circular mask:
    shape = fourier.shape
    H = high_pass_filter(shape)
    filtered_spectrum = fourier * H

    """ rows, cols = fourier.shape
    center = (int(cols / 2) , int(rows / 2))
    radius = int(0.1 * (min(rows,cols)/2))
    if radius == 0 :
        radius = 1 
    
    Y, X = np.ogrid[:rows, :cols]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)

    mask = dist_from_center <= radius
    masked_img = fourier.copy()
    masked_img[mask] = 0 #filter high values in middle of fourier plot
    #so the filtered spectrum is created:
    filtered_spectrum = masked_img """

    """ # Normalize, gamma correct, and rescale the high-passed spectrum
    PH_abs = np.abs(filtered_spectrum)
    PH_max = np.max(PH_abs)
    PH_normalized = (PH_abs / PH_max) ** 4 * PH_max """

    #Scale to fill range [0, 1], to make peaks more visual
    magnitude_spectrum = np.abs(filtered_spectrum)
    scaled_spectrum = (magnitude_spectrum - magnitude_spectrum.min()) / (magnitude_spectrum.max() - magnitude_spectrum.min())

    #Apply gamma correction with exponent 0.5
    gamma_corrected_spectrum = np.power(scaled_spectrum, 4)* magnitude_spectrum.max()

    return gamma_corrected_spectrum



#main script:

testcases = ["example.jpg"]

for testcase in testcases:
    #define constants
    a = np.random.rand(8) #initial model parameters: a 3x3 filter with a central value assumed to be 0
    a = a / a.sum() #make 3x3 filter unit-sum
    s = 0.005 #initial standard deviation
    d = 0.1 #outlier model probability

    #Read and normalize image into range [0,1] greyscale
    #To analyze an image for traces of re-sampling, the image must be to gray-scale. The reason for this conversion is that the individual
    #color channels may contain artifacts from color filter array interpolation and these artifacts could confound the analysis
    I = cv2.imread(testcase, cv2.IMREAD_GRAYSCALE)

    #normalize spectrum
    I = I - I.min()
    I = I/I.max()

    #build matrices for weighted least-squares
    #For high resolution images, the matrix F and vector f may become
    #prohibitively large to manage in memory. Thus a smaller random subset of pixels will always be used to
    #estimate the model parameters. Once the model parameters have been estimated, the probability for
    #every pixel can be computed
    F, f = build_matrices(I)

    #expectation-maximization loop
    #E-step and the M-step are iteratively executed until a stable estimate of a is achieved
    c = 0

    #the likelihood that the sample at location x,y is interpolated from its neighbours:
    w = np.zeros(F.shape[0])

    #estimate model parameters
    while c < 100:
        # E-step
        s2 = 0
        k = 0
        
        for y in range(1, I.shape[0] - 1):
            for x in range(1, I.shape[1] - 1):
                #calculate probability for every pixel
                r = compute_residual(a, I, x, y)
                g = np.exp(-r**2 / s) #risidual error -> probability
                w[k] = g / (g + d) #probability that sample k belongs to model
                s2 = s2 + w[k] * r**2 #accumulate for new standard deviation
                k = k + 1

        s = s2 / w.sum() #update standard deviation

        #M-step
        #trying to construct a diagonal weighting matrix that is very big will result in memory issues in python
        #hence we must do calculations with the vector directly;
        #Effect of multiplying a matrix by a diagonal matrix = multiply each column of F.T by the corresponding element in w
        #achieved with: F.T * w (* is element wise, @ is matrix multiplication in python)
        #from the book: 
        #W = diag(w) // diagonal weighting matrix with w along the diagonal
        #a2 = (F.T W.T W F)^-1 F.T W.T W f;
        #So the new model parameter will be calculated like so, using the vector directly in calculations:
        a2 = np.linalg.inv(F.T * w * w @ F) @ F.T * w * w @ f

        #check stopping conditions
        if np.linalg.norm(a - a2) < 0.01:
            break #done
        else:
            a = a2
            c = c + 1

    #show probability map
    #w is a probability vector that needs to be reshaped to an image first
    wReshaped = w.reshape(I.shape[0]-2,I.shape[1]-2)

    #save probabilitymap seperately?

    #calculate fourrier transforms of region of interest to confrim periodic patterns or distinguish from uniform regions

    #ask user to select pixel pairs to identify regions:
    fig, ax = plt.subplots()
    ax.imshow(wReshaped, cmap='gray', vmin=0, vmax=1)
    fig.canvas.mpl_connect('button_press_event', on_click)
    plt.show()

    print(f'You selected {len(selected_points)} points: {selected_points}')

    #calculate fourier transforms of regions + show image plots
    #first plot original probabilitymap
    fig, ax = plt.subplots(1+int(len(selected_points)/2),2)
    ax[0,0].imshow(wReshaped,cmap="gray", vmin=0, vmax=1)
    ax[0,0].axis('off')
    ax[0,1].imshow(calculate_fourier_map(wReshaped),cmap="gray", vmin=0, vmax=1)
    ax[0,1].axis('off')

    #Loop through the selected points in steps of 2
    for i in range(0, len(selected_points), 2):
        if i + 1 < len(selected_points): #if the user selected an uneven number of point, ignore last point
            (x1, y1) = selected_points[i]
            (x2, y2) = selected_points[i + 1]
            
            #extract the part between the selected points
            wReshaped_part = wReshaped[min(y1, y2):max(y1, y2)+1, min(x1, x2):max(x1, x2)+1]
            
            #calculate fourier for extracted part
            four_region = calculate_fourier_map(wReshaped_part)

            #plot fourier allongside highlighted region prob map
            #highligth selected zone:
            highlighted = wReshaped.copy()
            highlighted[min(y1,y2):max(y1,y2)+1,[x1,(x1+1),(x1-1),x2,(x2+1),(x2-1)]]=0 #y-stripes
            highlighted[[y1,(y1-1),(y1+1),y2,(y2-1),(y2+1)],min(x1,x2):max(x1,x2)+1 ]=0 #x-stripes
            ax[int((i/2)+1),0].imshow(highlighted,cmap="gray", vmin=0, vmax=1)
            ax[int((i/2)+1),0].axis('off')
            ax[int((i/2)+1),1].imshow(four_region,cmap="gray", vmin=0, vmax=1)
            ax[int((i/2)+1),1].axis('off')

    selected_points = []
    #plt.imshow(wReshaped,cmap='gray', vmin=0, vmax=1)
    #plt.axis('off')
    #plt.savefig('resampled_ex1_fake_3x3_st.png')#standard 640 x 480
    plt.savefig(testcase + '.png', dpi=200)#increase quality plot

    plt.show()


""" 
    This file was used for development and early testing of the noise wavelet blocking algorithm pioneered by:
    Babak Mahdian & Stanislav Saic - "Using noise inconsistencies for blind image forensics"
    A block merging step has not been included because all attempts yielded unsastifactory results and made analysis more difficult.

    In the paper the merging step appears highly effective, but this could be because of the isolated test conditions where the only obserable
    noise in the image was the one added by the researchers for testing puposes.

    What it does:
    The code reads an image inside the execution map, called example.jpg.
    The code then saves this image in 2 JPEG qualities: 60 and 90
    A splice is made where a central part of compression 60 is inserted into 90
    this splice is also saved as "splice_no_offset.jpg"
    Then the code calculates the ghostmaps from quaility qmin to qmax in steps of 5
    display the ghostmaps per quality
    saves the ghostmaps as "example_ghostmaps.png"

    You can fill in offset x and y to target a specific offset.

    How to use:
    install dependencies: pip install -r requirements_noise.txt
    run script: python noise_wavelets.py
"""

import numpy as np
import pywt
import matplotlib.pyplot as plt
import cv2

def get_noise_map(impath):
    blocksize = 8

    im = cv2.imread(impath, cv2.IMREAD_GRAYSCALE)
    y = np.double(im)

    #3.1 wavelet transform
    cA1, (cH, cV, cD) = pywt.dwt2(y, 'db8')
    cD = cD[:cD.shape[0] // blocksize * blocksize, :cD.shape[1] // blocksize * blocksize]
    
    #3.2 non overlapping blocks segmentation
    block = np.zeros((cD.shape[0] // blocksize, cD.shape[1] // blocksize, blocksize ** 2))
    
    for ii in range(0, cD.shape[0] - blocksize + 1, blocksize):
        for jj in range(0, cD.shape[1] - blocksize + 1, blocksize):
            block_elements = cD[ii:ii+blocksize, jj:jj+blocksize]
            block[ii // blocksize, jj // blocksize, :] = block_elements.flatten()
    
    #3.3 noise level estimation
    noise_map = np.median(np.abs(block), axis=2) / 0.6745

    #3.4 blocks merging - 
    #not included, merging results for real images were dissatisfactory, merging works better in lab conditions

    return noise_map


#main script
testcases = ["example.jpg"]

for testcase in testcases:
    noisemap = get_noise_map(testcase)
    #rescale the image to span the full spectrum (0-255)
    noisemap = cv2.normalize(noisemap, None, 0, 255, cv2.NORM_MINMAX)

    #second map
    originalBGR = cv2.imread(testcase)
    #change BGR to RGB for matplotlib
    originalRGB = cv2.cvtColor(originalBGR, cv2.COLOR_BGR2RGB) 
    originalRGB_normalized = originalRGB.astype(np.float32) / 255.0
    plt.clf()
    plt.subplot(1,2,1)
    plt.imshow(originalRGB_normalized)
    #plt.title('Original Image')
    plt.axis('off')
    #add map:
    plt.subplot(1,2,2)
    plt.imshow(noisemap, cmap='gray')
    plt.axis('off')
    plt.draw()

    noisemap_2_filepath = (testcase + 'output_grayscaleblok8.png')
    plt.savefig(noisemap_2_filepath, dpi=200)
    


""" 
    This file was used for development and early testing of the JPEG GHOST algorithm, as described in 
    "Exposing Digital Forgeries from JPEG Ghosts" by Hany Farid.  
    The book "Digital Image Forensics" by Hany Farid also provides insight into how this technique is able 
    to localize manipulations when two parts of an image are of different quality.

    Clarative comments have been added to the code based on the mentioned resources.

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
    install dependencies: pip install -r requirements_ghost.txt
    run script: python ghostmmaps.py
 """

import matplotlib.pyplot as plt
import math
import numpy as np
import cv2

Qmin = 60
Qmax = 90
Qstep = 5
averagingBlock = 16


#load original
original = np.double(cv2.imread("example.jpg"))
ydim, xdim, zdim = original.shape

#save with different qualities, 90 & 60
cv2.imwrite('compr90.jpg', original, [cv2.IMWRITE_JPEG_QUALITY, 90])
cv2.imwrite('compr60.jpg', original, [cv2.IMWRITE_JPEG_QUALITY, 60])

#new image as combination of resaved imaged
compressed90 = np.double(cv2.imread("compr90.jpg"))
compressed60 = np.double(cv2.imread("compr60.jpg"))
xrng = slice(round(xdim/2 - 500), round(xdim/2 + 500)) #altered regions
yrng = slice(round(ydim/2 - 500), round(ydim/2 + 500))
xrngSh = slice(round((xdim/2 - 500)), round((xdim/2 + 500))) #shifted regions
yrngSH = slice(round((ydim/2 - 500)), round((ydim/2 + 500)))
compressed90[yrng, xrng, :] = compressed60[yrngSH, xrngSh, :] #splice creation
cv2.imwrite('splice_no_offset.jpg', compressed90, [cv2.IMWRITE_JPEG_QUALITY, 90])

splice = np.double(cv2.imread("splice_no_offset.jpg"))

#construct ghostmap
nQ = int((Qmax-Qmin)/Qstep)+1
ghostmaps = np.zeros((ydim, xdim, nQ))
i=0

#compute difference splice and re-compressed versions (at given quality levels) of splice
for quality in range(Qmin, Qmax+1, Qstep):

    shifted_original = np.roll(splice, 0, axis=1)#offset x
    shifted_original = np.roll(shifted_original, 0, axis=0)#offset y
    
    #compress image to new quality
    tempvar1 = cv2.imencode('.jpg', shifted_original, [int(cv2.IMWRITE_JPEG_QUALITY), quality])[1].tobytes()
    tempcar2 = np.frombuffer(tempvar1, np.byte)
    tmpResave = np.double(cv2.imdecode(tempcar2, cv2.IMREAD_ANYCOLOR))

    #compute difference original and compressed version and average over RGB, in part because:
    #Different cameras and photo-editing software packages will employ different JPEG quality scales and quantization tables
    #Becasue we average over a block size of 16, the difference images are computed by averaging across all spatial frequencies. 
    #As a result, small differences in the original and quantization tables from our recompressed version
    #will not likely have a significant impact
    for z in range(zdim):
        ghostmaps[:, :, i] += np.square(shifted_original[:, :, z].astype(np.double) - tmpResave[:, :, z])
    
    ghostmaps[:, :, i] /= zdim
    i +=1


#compute average over larger area to counter complicating factor:
#Because the image difference is computed across
#all spatial frequencies, a region with small amounts of high
#spatial frequency content (e.g., a mostly uniform sky) will have
#a lower difference as compared to a highly textured region
#(e.g., grass). In order to compensate for these differences,
#we consider a spatially averaged and normalized difference measure.
blkE = np.zeros((int((ydim) / averagingBlock), int((xdim) / averagingBlock), nQ))
for c in range(nQ):
    cy = 0
    for y in range(0, ydim - averagingBlock, averagingBlock):
        cx = 0
        for x in range(0, xdim - averagingBlock, averagingBlock):
            bE = ghostmaps[y:y + averagingBlock, x:x + averagingBlock, c]
            blkE[cy, cx, c] = np.mean(bE)
            cx += 1
        cy += 1


#normalize difference, enhances contrast for inspection
minval = np.min(blkE, axis=2)
maxval = np.max(blkE, axis=2)
for c in range(nQ):
    blkE[:, :, c] = (blkE[:, :, c] - minval) / (maxval - minval)



sp = math.ceil(math.sqrt(nQ+1))
#plot original image - needs to be normalized first for a subplot & converted to RGB, because cv2 works by default on BGR
original_uint8 = cv2.convertScaleAbs(splice) #cv2.cvtColor expects input images to have depth of 8-bit per channel
original_rgb = cv2.cvtColor(original_uint8, cv2.COLOR_BGR2RGB)
originalRGB_normalized = original_rgb.astype(np.float32) / 255.0
plt.subplot(sp, sp, 1)
plt.imshow(originalRGB_normalized)
plt.title('Forged')
plt.axis('off')
#add maps:
for c in range(nQ):
    plt.subplot(sp, sp, c+2)
    plt.imshow(blkE[:, :, c],cmap='gray', vmin=0, vmax=1)
    plt.axis('off')
    plt.title(str(Qmin+c*Qstep))
    plt.draw()

#plt.rc('figure', titlesize=16)  # fontsize of the figure title
plt.suptitle('Ghost plots offset x = 0 and y = 0')  
plt.savefig('example_ghostmaps.png', dpi=200)

plt.show()


import run
import scipy.misc
import matplotlib.pyplot as plt
import numpy as np
import skimage.feature
import skimage.color
import learn

im = scipy.misc.imread("sampleData/images/ADE_val_00000003.jpg")
ima = scipy.misc.imread("sampleData/annotations/ADE_val_00000003.png")
imc = skimage.feature.canny(skimage.color.rgb2gray(im),sigma=1.0)
cd = learn.read_distf("sampled_local_1_normalized.mat")
fd = learn.read_distf("sampled_color_normalized.mat")

scores = run.run_nn_full(im)
imnn = np.argmax(scores,axis=2)

NUM_ITERS = 5

#print np.shape(imnn)
#print len(imnn)
#print np.shape(scores)
#print imnn
imsample = imnn
for _ in range(NUM_ITERS):
    imsample = run.sample(imsample,scores,imc,im,cd,fd)
#print imsample

plt.subplot(2,1,1)
run.display_annotation(imsample)

plt.subplot(2,1,2)
run.display_annotation(ima)

plt.show()


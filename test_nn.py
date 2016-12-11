import run
import scipy.misc
import matplotlib.pyplot as plt
import numpy as np

im = scipy.misc.imread("sampleData/images/ADE_val_00000003.jpg")
ima = scipy.misc.imread("sampleData/annotations/ADE_val_00000003.png")

imnn = run.run_nn(im)

#imnn = run.run_nn_full(im)

print np.sum(imnn == ima),"/",len(ima) * len(ima[0])

plt.subplot(2,1,1)
run.display_annotation(imnn)

plt.subplot(2,1,2)
run.display_annotation(ima - 1)

plt.show()


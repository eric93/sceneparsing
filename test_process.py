import scipy.misc
import learn
import numpy as np

dist = np.copy(learn.default)
learn.process_image(dist,
        scipy.misc.imread("sampleData/annotations/ADE_val_00000003.png"),
        scipy.misc.imread("sampleData/images/ADE_val_00000003.jpg"))
learn.write_dist(dist,"test_exact")

import scipy.misc
import learn
import numpy as np

annfilenames = ["ADEChallengeData2016/annotations/training/ADE_train_" + ("0" * (8 - len(str(x)))) + str(x) + ".png" for x in range(1,20211)]
imgfilenames = ["ADEChallengeData2016/images/training/ADE_train_" + ("0" * (8 - len(str(x)))) + str(x) + ".jpg" for x in range(1,20211)]
dist = np.copy(learn.default)
print "Analyzing..."
learn.process_all_local_canny(dist, annfilenames, imgfilenames)
print "Done."
learn.write_dist(dist,"sampled_local_3_full")

import caffe
import skimage.transform
import numpy as np
import numpy.random as rand
import random
import matplotlib.pyplot as plt
import scipy.misc
import learn

network = caffe.Net("models/deploy_FCN.prototxt", "FCN_iter_160000.caffemodel", caffe.TEST)

def rescale_scores(scores,new_w,new_h):
    old_w = len(scores)
    old_h = len(scores[0])
    ret = np.zeros((new_w,new_h,len(scores[0,0])),dtype=np.float32)
    for x in range(new_w):
        for y in range(new_h):
            old_x = np.floor(x * (old_w / float(new_w)))
            old_y = np.floor(y * (old_h / float(new_h)))
            ret[x,y] = scores[old_x,old_y]
    return ret

def run_nn(im):
    im_rs = skimage.transform.resize(im,(384,384),preserve_range=True)
    im_bgr = im_rs[:,:,::-1]
    im_centered = np.stack((im_bgr[:,:,0]-109.5388,im_bgr[:,:,1]-118.6897,im_bgr[:,:,2]-124.6901),axis=2)
    im_perm = np.transpose(im_centered,axes=[2,0,1])
    network.blobs['data'].data[...] = np.array([im_perm])
    score = network.forward()['score']
    labeled_img = np.argmax(rescale_scores(np.transpose(score[0],[1,2,0]),len(im),len(im[0])),axis=2)
    return labeled_img

def run_nn_full(im):
    im_rs = skimage.transform.resize(im,(384,384),preserve_range=True)
    im_bgr = im_rs[:,:,::-1]
    im_centered = np.stack((im_bgr[:,:,0]-109.5388,im_bgr[:,:,1]-118.6897,im_bgr[:,:,2]-124.6901),axis=2)
    im_perm = np.transpose(im_centered,axes=[2,0,1])
    network.blobs['data'].data[...] = np.array([im_perm])
    score = network.forward()['score']
    labeled_img = np.transpose(score[0],[1,2,0])
    return rescale_scores(labeled_img,len(im),len(im[0]))


colormap_raw = dict()
def colormap(key):
    if key in colormap_raw.keys():
        return colormap_raw[key]
    else:
        levels = range(32,256,32)
        color = tuple(random.choice(levels) for _ in range(3))
        colormap_raw[key] = color
        return color

SAMPLES_PER_POINT = 10

def log_softmax(vec):
        return vec - scipy.misc.logsumexp(vec)

def softmax(vec):
        return np.exp(log_softmax(vec))

def sample(prev_sample, scores, imc, dist):
    new_sample = prev_sample[:,:]
    for x in range(len(prev_sample)):
        for y in range(len(prev_sample[0])):
            score = softmax(scores[x][y])
            for _ in range(SAMPLES_PER_POINT):
                xn,yn = 0,0
                done = False
                while (not done) and (xn == x or yn == y):
                    #xn = random.randrange(max(0,x-10),min(x+10,len(prev_sample)))
                    #yn = random.randrange(max(0,y-10),min(y+10,len(prev_sample[0])))
                    xn = random.randrange(len(prev_sample))
                    yn = random.randrange(len(prev_sample[0]))

                b = learn.get_bucket([x,y],[xn,yn])
                e = learn.crosses_edge(x,y,xn,yn,imc)
                score *= dist[e][b][prev_sample[x][y]][:]
                #print score
                sum_s = np.sum(score)
                score = score / sum_s
                #print score
                new_sample[x][y] = rand.choice(len(score),None,p=score)
    return new_sample

def sample_color(prev_sample, scores, imo, dist):
    new_sample = prev_sample[:,:]
    for x in range(len(prev_sample)):
        for y in range(len(prev_sample[0])):
            score = softmax(scores[x][y])
            for _ in range(SAMPLES_PER_POINT):
                xn,yn = 0,0
                done = False
                while (not done) and (xn == x or yn == y):
                    #xn = random.randrange(max(0,x-10),min(x+10,len(prev_sample)))
                    #yn = random.randrange(max(0,y-10),min(y+10,len(prev_sample[0])))
                    xn = random.randrange(len(prev_sample))
                    yn = random.randrange(len(prev_sample[0]))

                b = learn.get_bucket([x,y],[xn,yn])

                c1 = imo[x,y]
                c2 = imo[xn,yn]
                c = learn.similar_color(c1,c2)
                score *= dist[c][b][prev_sample[x][y]][:]
                #print score
                sum_s = np.sum(score)
                score = score / sum_s
                #print score
                new_sample[x][y] = rand.choice(len(score),None,p=score)
    return new_sample


def display_annotation(ima):
    new_img = np.zeros((len(ima),len(ima[0]),3),np.uint8)
    for x in range(len(ima)):
        for y in range(len(ima[0])):
            new_img[x][y] = colormap(ima[x][y])

    plt.imshow(new_img)

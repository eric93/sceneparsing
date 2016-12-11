import matplotlib.pyplot as plt
import numpy as np
import random
import scipy.spatial.distance
import scipy.io
import scipy.misc
import math
import skimage.color
import skimage.feature

colormap_raw = {}
def colormap(c):
    if c in colormap_raw:
        return colormap_raw[c]
    else:
        levels = range(32,256,32)
        value = tuple(random.choice(levels) for _ in range(3))
        colormap_raw[c] = np.array(value)
        return value

def displayseg(im):
    im2 = np.zeros((len(im),len(im[0]),3), dtype=np.uint8)
    for i in range(len(im2)):
        for j in range(len(im2[0])):
            im2[i][j] = colormap(im[i][j])

    plt.imshow(im2)
    plt.show()


NUM_CATEGORIES = 151
BUCKETS = [10.0,50.0,75.0,100.0,200.0,500.0]
COLOR_EPSILON = 10.0
CANNY_SIGMA=3.0
SAMPLES_PER_IMAGE = 10000
SAMPLES = 100000

def get_bucket(p1,p2):
    d = scipy.spatial.distance.euclidean(p1,p2)
    for i in range(len(BUCKETS)):
        if d < BUCKETS[i]:
            return i

    return len(BUCKETS)

default = np.ones((2, len(BUCKETS) + 1, NUM_CATEGORIES, NUM_CATEGORIES), dtype = np.int32)

def process_image(dist_handle,ima,imo):
    W = len(ima)
    H = len(ima[0])
    for x1 in range(W):
        for y1 in range(H):
            #print x1,",",y1
            for x2 in range(max(x1-5,0),min(x1+5,W)):
                for y2 in range(max(y1-5,0),min(y1+5,H)):
                    #dist_handle[0][0][ima[x1,y1]][ima[x2,y2]] += 1
                    #d = scipy.spatial.distance.euclidean([x1,y1],[x2,y2])

                    dist_handle[0][1][ima[x1,y1]][ima[x2,y2]] += 1



def crosses_edge(x1,y1,x2,y2,imc):
    delta_x = x2 - x1
    delta_y = y2 - y1

    def imc_edge(x,y):
        if x < 0 or x >= len(imc) or y < 0 or y >= len(imc[0]):
            return False
        else:
            return True

    if delta_x == 0 and delta_y == 0:
        return False

    if abs(delta_x) > abs(delta_y):
        m = float(delta_y) / delta_x
        yint = y1 - (m * x1)
        if x1 < x2:
            for xt in range(x1,x2):
                yt = m * xt + yint
                if imc_edge(xt,math.floor(yt)) or imc_edge(xt,math.ceil(yt)):
                    return True
        else:
            for xt in range(x2,x1):
                yt = m * xt + yint
                if imc_edge(xt,math.floor(yt)) or imc_edge(xt,math.ceil(yt)):
                    return True

    else:
        m = float(delta_x) / delta_y
        xint = x1 - (m * y1)
        if y1 < y2:
            for yt in range(y1,y2):
                xt = m * yt + xint
                if imc_edge(math.floor(xt),yt) or imc_edge(math.ceil(xt),yt):
                    return True
        else:
            for yt in range(y2,y1):
                xt = m * yt + xint
                if imc_edge(math.floor(xt),yt) or imc_edge(math.ceil(xt),yt):
                    return True
    return False


def process_all_local_canny(dist_handle,annfilenames, imgfilenames):
    for i in range(len(annfilenames)):
        if i % 50 == 0:
                print i

        imgnum = random.randrange(len(annfilenames))
        ima = scipy.misc.imread(annfilenames[imgnum])
        imo = scipy.misc.imread(imgfilenames[imgnum])
        imc = skimage.feature.canny(skimage.color.rgb2gray(imo),sigma=CANNY_SIGMA)

        for j in range(SAMPLES_PER_IMAGE):
            x1,y1,x2,y2 = 0,0,0,0
            while (x1 == y1) or (x2 == y2) or scipy.spatial.distance.euclidean([x1,y1],[x2,y2]) > 10.0:
            #while (x1 == y1) or (x2 == y2):
                x1 = random.randrange(len(ima))
                x2 = random.randrange(max(0,x1-10),min(x1+10,len(ima)))
                y1 = random.randrange(len(ima[0]))
                y2 = random.randrange(max(0,y1-10),min(y1+10,len(ima[0])))

            p1 = np.array([x1,y1])
            p2 = np.array([x2,y2])

            e = crosses_edge(x1,y1,x2,y2,imc)

            dist_handle[1 if e else 0][0][ima[x1,y1]][ima[x2,y2]] += 1


def process_all_local_canny_2(dist_handle,annfilenames, imgfilenames):
    for i in range(len(annfilenames)):
        if i % 50 == 0:
                print i

        imgnum = random.randrange(len(annfilenames))
        ima = scipy.misc.imread(annfilenames[imgnum])
        imo = scipy.misc.imread(imgfilenames[imgnum])
        imc = skimage.feature.canny(skimage.color.rgb2gray(imo),sigma=CANNY_SIGMA)

        for j in range(SAMPLES_PER_IMAGE):
            x1,y1,x2,y2 = 0,0,0,0
            #while (x1 == y1) or (x2 == y2) or scipy.spatial.distance.euclidean([x1,y1],[x2,y2]) > 10.0:
            while (x1 == y1) or (x2 == y2):
                x1 = random.randrange(len(ima))
                x2 = random.randrange(max(0,x1-100),min(x1+100,len(ima)))
                y1 = random.randrange(len(ima[0]))
                y2 = random.randrange(max(0,y1-100),min(y1+100,len(ima[0])))

            p1 = np.array([x1,y1])
            p2 = np.array([x2,y2])

            e = crosses_edge(x1,y1,x2,y2,imc)

            b = get_bucket(p1,p2)
            dist_handle[1 if e else 0][b][ima[x1,y1]][ima[x2,y2]] += 1





def process_all_canny(dist_handle,annfilenames, imgfilenames):
    for i in range(len(annfilenames)):
        if i % 50 == 0:
                print i

        imgnum = random.randrange(len(annfilenames))
        ima = scipy.misc.imread(annfilenames[imgnum])
        imo = scipy.misc.imread(imgfilenames[imgnum])
        imc = skimage.feature.canny(skimage.color.rgb2gray(imo),sigma=CANNY_SIGMA)

        for j in range(SAMPLES_PER_IMAGE):
            x1,y1,x2,y2 = 0,0,0,0
            while (x1 == y1) or (x2 == y2):
                x1 = random.randrange(len(ima))
                x2 = random.randrange(len(ima))
                y1 = random.randrange(len(ima[0]))
                y2 = random.randrange(len(ima[0]))

            p1 = np.array([x1,y1])
            p2 = np.array([x2,y2])

            e = crosses_edge(x1,y1,x2,y2,imc)

            b = get_bucket(p1,p2)
            dist_handle[e][b][ima[x1,y1]][ima[x2,y2]] += 1

def similar_color(c1,c2):
    if scipy.spatial.distance.euclidean(c1,c2) < COLOR_EPSILON:
        return 0
    else:
        return 1


def process_all(dist_handle,annfilenames, imgfilenames):
    for i in range(len(annfilenames)):
        if i % 50 == 0:
                print i

        imgnum = random.randrange(len(annfilenames))
        ima = scipy.misc.imread(annfilenames[imgnum])
        imo = scipy.misc.imread(imgfilenames[imgnum])

        for j in range(SAMPLES_PER_IMAGE):
            x1,y1,x2,y2 = 0,0,0,0
            while (x1 == y1) or (x2 == y2):
                x1 = random.randrange(len(ima))
                x2 = random.randrange(len(ima))
                y1 = random.randrange(len(ima[0]))
                y2 = random.randrange(len(ima[0]))

            p1 = np.array([x1,y1])
            p2 = np.array([x2,y2])

            c1 = imo[x1,y1]
            c2 = imo[x2,y2]

            if scipy.spatial.distance.euclidean(c1,c2) < COLOR_EPSILON:
                c = 0
            else:
                c = 1


            b = get_bucket(p1,p2)
            dist_handle[c][b][ima[x1,y1]][ima[x2,y2]] += 1

def write_dist(dist,filename):
    m = dict()
    for c in range(2):
        for b in range(len(BUCKETS) + 1):
            typestr = "_similar" if c == 0 else "_different"
            m["bucket_" + str(b) + typestr] = dist[c][b]

    print m
    scipy.io.savemat(filename , m)

def read_dist(filename):
    new_dist = np.zeros((2,len(BUCKETS) + 1, NUM_CATEGORIES, NUM_CATEGORIES), dtype = np.uint32)
    m = scipy.io.loadmat(filename)
    for c in range(2):
        for b in range(len(BUCKETS) + 1):
            typestr = "_similar" if c == 0 else "_different"
            new_dist[c][b] = m["bucket_" + str(b) + typestr]
    return new_dist

def read_distf(filename):
    new_dist = np.zeros((2,len(BUCKETS) + 1, NUM_CATEGORIES, NUM_CATEGORIES), dtype = np.float32)
    m = scipy.io.loadmat(filename)
    for c in range(2):
        for b in range(len(BUCKETS) + 1):
            typestr = "_similar" if c == 0 else "_different"
            new_dist[c][b] = m["bucket_" + str(b) + typestr]
    return new_dist


def reshape_dist(dist):
    new_dist = np.zeros((2,len(BUCKETS) + 1, NUM_CATEGORIES, NUM_CATEGORIES), dtype = np.uint32)
    for c in range(2):
        for b in range(len(BUCKETS) + 1):
            for i in range(NUM_CATEGORIES):
                for j in range(NUM_CATEGORIES):
                    if i < j:
                        new_dist[c][b][i][j] = dist[c][b][i][j] + dist[c][b][j][i] - 1
                    elif i == j:
                        new_dist[c][b][i][j] = dist[c][b][i][j]

    return new_dist

def normalize_reshaped(dist):
    new_dist = np.zeros((2,len(BUCKETS) + 1, NUM_CATEGORIES, NUM_CATEGORIES), dtype = np.float32)
    for c in range(2):
        for b in range(len(BUCKETS) + 1):
            for i in range(NUM_CATEGORIES):
                s = 0
                for j in range(NUM_CATEGORIES):
                    if i < j:
                        s += dist[c][b][i][j]
                    else:
                        s += dist[c][b][j][i]
                print c,",",b,",",i,":",s
                for j in range(NUM_CATEGORIES):
                    new_dist[c][b][i][j] = dist[c][b][i][j] / float(s)
    return new_dist


def normalize(dist):
    new_dist = np.zeros((2, len(BUCKETS) + 1, NUM_CATEGORIES, NUM_CATEGORIES), dtype = np.float32)
    for c in range(2):
        for b in range(len(BUCKETS) + 1):
            for i in range(NUM_CATEGORIES):
                s = 0
                for j in range(NUM_CATEGORIES):
                    s += dist[c][b][i][j]

                for j in range(NUM_CATEGORIES):
                    new_dist[c][b][i][j] = (float(dist[c][b][i][j]) / float(s))

    return new_dist



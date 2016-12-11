import skimage.feature
import skimage.color
import scipy.misc
import matplotlib.pyplot as plt
import random
import math


imgfilenames = ["ADEChallengeData2016/images/training/ADE_train_" + ("0" * (8 - len(str(x)))) + str(x) + ".jpg" for x in range(1,20211)]
fnrand = random.choice(imgfilenames)

im = scipy.misc.imread(fnrand)
imbw = skimage.color.rgb2gray(im)
imc = skimage.feature.canny(imbw,sigma=4.0)

x1,y1,x2,y2 = 0,0,0,0
x1 = random.randrange(len(imc))
x2 = random.randrange(max(0,x1-10),min(x1+10,len(imc)))
y1 = random.randrange(len(imc[0]))
y2 = random.randrange(max(0,y1-10),min(y1+10,len(imc[0])))

print x1,",",y1,":",x2,",",y2

delta_x = x2 - x1
delta_y = y2 - y1

xline = []
yline = []
result = False
if delta_x > delta_y:
    m = float(delta_y) / delta_x
    yint = y1 - (m * x1)
    if x1 < x2:
        for xt in range(x1,x2):
            yt = m * xt + yint
            xline += [xt]
            yline += [math.floor(yt)]
            if imc[xt,math.floor(yt)] or imc[xt,math.ceil(yt)]:
                result = True
    else:
        for xt in range(x2,x1):
            yt = m * xt + yint
            xline += [xt]
            yline += [math.floor(yt)]
            if imc[xt,math.floor(yt)] or imc[xt,math.ceil(yt)]:
                result = True

else:
    m = float(delta_x) / delta_y
    xint = x1 - (m * y1)
    if y1 < y2:
        for yt in range(y1,y2):
            xt = m * yt + xint
            yline += [yt]
            xline += [math.floor(xt)]
            #xline += [math.ceil(xt)]
            if imc[math.floor(xt),yt] or imc[math.ceil(xt),yt]:
                result = True
    else:
        for yt in range(y2,y1):
            xt = m * yt + xint
            yline += [yt]
            xline += [math.floor(xt)]
            #xline += [math.ceil(xt)]
            if imc[math.floor(xt),yt] or imc[math.ceil(xt),yt]:
                result = True

print result
plt.subplot(1,2,2)
ax=plt.imshow(imc)
plt.plot(yline,xline,marker='o')
plt.subplot(1,2,1)
plt.imshow(im)
plt.show()

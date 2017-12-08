import numpy as np
import cv2
import os, random, sys
from PIL import Image
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from skimage.measure import LineModelND, ransac
from skimage.color import rgb2hsv
from scipy.spatial import distance
from sklearn.cluster import MiniBatchKMeans
import datetime
import itertools
import time
import copy

print(time.ctime())

GENERATE_OR_SEGMENT = 0

#ROOT_DIR = '../data/UERCTest'
ROOT_DIR = '../data/RESULTS/UERC_org'
RESULTS_DIR = '../data/RESULTS/HAIR'
ER_SOURCE = '../data/data/hair'
MASK_SOURCE = '../data/uerc/Results/Truth' # only used for UERCTest generation
img_size = 227


def medianColor(frame, img):
    (x, y, h, w) = frame
    medRow = np.median(img[y + (h / 8):y + (h - (h / 8)), x + (w / 8):x + (w - (w / 8))], axis=0);
    return np.median(medRow, axis=0)


def medianColors(frames, img):
    colors = np.array(medianColor(frames[0, :], img));
    i = 0
    for (x, y, h, w) in frames[1:, :]:
        col = np.median(np.median(
            img[y + (h / 8):y + (h - (h / 8)), x + (w / 8):x + (w - (w / 8))], axis=0), axis=0)
        colors = np.vstack((colors, col))
        i += 1
    return colors


def adjust_gamma(img, gamma=1.0):
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(img, table)


def xorVec(vec):
    a = False
    for x in vec:
        if(x):
            a = True
            break

    return a


def segment(img, hCuts, wCuts, grabcutIter=4, ransacThreashold=24, preGamma=1.0, postGamma=1.5, showGraph=False, useFullImageRes=False, earMask=None):

    #img = cv2.imread(imgName)

    height, width = img.shape[:2]

    mask = cv2.GC_PR_FGD * np.ones((height, width), dtype='uint8')

    img2 = adjust_gamma(img, preGamma)

    if(useFullImageRes):
        col = np.reshape(img2, (height * width, 3)).astype('float')
        if earMask is not None:
            eM = np.reshape(earMask, (height * width)).astype('float') == 1
            col = col[eM]
    else:
        # compute image frames
        x = np.linspace(0, width, wCuts)

        y = np.linspace(0, height, hCuts)
        frames = np.ndarray(((len(x) - 1) * (len(y) - 1), 4), dtype=int)

        for i in range(len(x) - 1):
            for j in range(len(y) - 1):
                frames[i * (len(y) - 1) + j, :] = [math.floor(x[i]), math.floor(y[j]), math.floor(
                    y[j + 1]) - math.floor(y[j]), math.floor(x[i + 1]) - math.floor(x[i])]

        col = medianColors(frames, img2)

    model_robust, inliers = ransac(
        col, LineModelND, min_samples=2, residual_threshold=ransacThreashold, max_trials=1000)

    outliers = inliers == False

    if(showGraph):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(col[inliers][:, 0], col[inliers][:, 1], col[inliers][:, 2], c='b',
                   marker='o', label='Inlier data')
        ax.scatter(col[outliers][:, 0], col[outliers][:, 1], col[outliers][:, 2], c='r',
                   marker='o', label='Outlier data')
        ax.legend(loc='lower left')
        ax.set_xlabel('Blue')
        ax.set_ylabel('Green')
        ax.set_zlabel('Red')
        plt.show()

    if(xorVec(outliers)):
        if (useFullImageRes):
            tmp = np.reshape(outliers, (height, width))
            mask[tmp] = cv2.GC_PR_BGD
        else:
            for (x, y, h, w) in frames[outliers]:
                mask[y:y + h, x:x + w] = cv2.GC_PR_BGD;
                #img[y:y+h,x:x+w,:] = [0, 0, 0]

        img = adjust_gamma(img, postGamma)

        # img4 = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        # plt.imshow(mask, cmap='gray')
        # plt.show()

        bgdModel = np.zeros((1, 65), np.float64)
        fgdModel = np.zeros((1, 65), np.float64)
        cv2.grabCut(img.astype('uint8'), mask, None, bgdModel,
                    fgdModel, grabcutIter, cv2.GC_INIT_WITH_MASK)

    mask = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.erode(mask, kernel, iterations=2)

    img = img * mask[:, :, np.newaxis]

    cv2.imwrite('results.jpg', img)

    return 1 - mask


def generateER(root, cla, imgName):
    if not imgName.endswith('.png'):
        return 0
    # load original image
    img = cv2.imread(root + '/' + cla + '/' + imgName)
    #img = cv2.resize(img, (img_size, img_size))
    imgOrig = copy.deepcopy(img)

    # add alpha to orig image
    #img = np.dstack( ( img, np.ones((img.shape[0],img.shape[1]))*255 ) )

    # randomly select one earring template
    choice = random.choice(os.listdir(ER_SOURCE))
    while choice.startswith('.'):
        choice = random.choice(os.listdir(ER_SOURCE))
    print choice
    er = cv2.imread(ER_SOURCE + '/' + choice, -1)


    # Headphones
    if choice.startswith('2'):

        mmsize = img.shape[0] if img.shape[0] < img.shape[1] else img.shape[1]
        newSize = random.randint(round(0.2 * mmsize), round(0.4 * mmsize))
        if not newSize%2 == 0:
            newSize = newSize + 1

        er = cv2.resize(er, (newSize, newSize))

        print 'headphones'
        x = (img.shape[0]/2)
        print 'x: ',x
        y = (img.shape[1]/2)
        print 'y: ',y
        er_x = (er.shape[0]/2)
        print 'er_x: ',er_x
        er_y = (er.shape[1]/2)
        print 'er_y: ',er_y
        for c in range(0,3):
           img[x-er_x:x+er_x, y-er_y:y+er_y, c] = img[x-er_x:x+er_x, y-er_y:y+er_y, c] * (1.0 - er[:,:,3]/255.0) + er[:,:,c] * (er[:,:,3]/255.0)

        # generate mask
        blank = np.zeros((img.shape[0], img.shape[1]))
        erFlat = np.zeros((er.shape[0], er.shape[1]))
        erFlat[er[:, :, 3] > 0] = 255
        #TODO mirar pq no la posa a lloc
        blank[x:x + erFlat.shape[0], y:y + erFlat.shape[1]] = erFlat[:, :] * (erFlat[:, :] / 255.0) + blank[x:x + erFlat.shape[0],
                                            y:y + erFlat.shape[1]] * (1.0 - erFlat[:, :] / 255.0)
        mask = 255 - blank


    # Hair
    elif choice.startswith('3'):
        print 'hair'
        x = img.shape[1]
        y = img.shape[0]
        print 'Img X: ',x,'  Img Y: ',y
        y_move = int(round(2*y/5))
        print 'y_move: ',y_move
        er = cv2.resize(er,(y_move,x))
        cv2.imshow('er',er)
        print 'hair new X: ',er.shape[1], '   hair new Y: ',er.shape[0]
        print 'img X: ',img.shape[1], '   img Y: ', img.shape[0]

        for c in range(0,3):
            img[0:x, 0:y_move, c] = img[0:x, 0:y_move, c] * (1.0 - er[:,:,3]/255.0) + er[:,:,c] * (er[:,:,3]/255.0)

        # generate mask
        blank = np.zeros((img.shape[0], img.shape[1]))
        erFlat = np.zeros((er.shape[0], er.shape[1]))
        erFlat[er[:, :, 3] > 0] = 255
        blank[0:erFlat.shape[0], 0:erFlat.shape[1]] = erFlat[:, :] * (erFlat[:, :] / 255.0) + blank[0: + erFlat.shape[0], 0: + erFlat.shape[1]] * (1.0 - erFlat[:, :] / 255.0)
        mask = 255 - blank

    # Earring
    else:
        # randomly resize the earring
        mmsize = img.shape[0] if img.shape[0] < img.shape[1] else img.shape[1]
        newSize = random.randint(round(0.2 * mmsize), round(0.4 * mmsize))

        er = cv2.resize(er, (newSize, newSize))

        # randomly change color of the earring
        colorShift = 100;
        er = np.int16(er)

        er[:,:,0] += random.randint(-colorShift, colorShift)
        er[:,:,1] += random.randint(-colorShift, colorShift)
        er[:,:,2] += random.randint(-colorShift, colorShift)

        er[er > 255] = 255
        er[er < 0] = 0
        er = np.uint8(er)

        # randomly move the earring
        xLeft = 0
        xRight = img.shape[0] - er.shape[0]
        yTop = round(img.shape[1]/2)
        yBottom = img.shape[1] - er.shape[1]
        x_offset = random.randint(xLeft, xRight)
        y_offset = random.randint(yTop, yBottom)
        # paste earring to the orig image
        for c in range(0,3):
            img[x_offset:x_offset+er.shape[0], y_offset:y_offset+er.shape[1], c] = er[:,:,c] * (er[:,:,3]/255.0) + img[x_offset:x_offset+er.shape[0], y_offset:y_offset+er.shape[1], c] * (1.0 - er[:,:,3]/255.0)

        #generate mask
        blank = np.zeros((img.shape[0], img.shape[1]))
        erFlat = np.zeros((er.shape[0], er.shape[1]))
        erFlat[er[:,:,3] > 0] = 255
        blank[x_offset:x_offset+erFlat.shape[0], y_offset:y_offset+erFlat.shape[1]] = erFlat[:,:] * (erFlat[:,:]/255.0) +  blank[x_offset:x_offset+erFlat.shape[0], y_offset:y_offset+erFlat.shape[1]] * (1.0 - erFlat[:,:]/255.0)
        mask = 255 - blank

    # if there is existing Mask add it to the currently generated mask
    if os.path.isfile(MASK_SOURCE + '/' + cla + '/' + imgName):
        print('COMBINING MASKS')
        eMask = cv2.imread(MASK_SOURCE + '/' + cla + '/' + imgName, 0)
        #eMask = cv2.resize(eMask, (img_size, img_size))
        mask[eMask == 0] = 0

    #print(os.path.basename(ROOT_DIR))
    writeResults(os.path.basename(ROOT_DIR), cla, imgName, img, mask, False, imgOrig)


def segmentER(root, cla, imgName):	
    # load original image
    img = cv2.imread(root + '/' + cla + '/' + imgName)
    #img = cv2.resize(img, (img_size, img_size))

    param = [2, 15, 0.3, 1.1]
    try:
        mask = segment(img, 28 * 5, 22 * 5, grabcutIter=param[0], ransacThreashold=param[1], preGamma=param[2], postGamma= param[3], showGraph=False, useFullImageRes=True)
        #sys.exit()
        loMask = mask == 0
        mask[loMask] = 255
        hiMask = mask < 255
        mask[hiMask] = 0
    except:
        print('Errroe! Mask will be ones.')
        mask = np.ones((img.shape[0], img.shape[1])) * 255
        print(mask.shape)

    writeResults(os.path.basename(ROOT_DIR), cla, imgName, img, mask, False, img)


def writeResults(rootName, cla, imgName, img, mask, isDetected, original):
    # join image and mask into a joint image with alpha channel
# joint = np.dstack( ( img, mask ) )

    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_imageOrig = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    joint = np.dstack( ( gray_image, gray_image, mask ) )

    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)
    root = RESULTS_DIR + '/' + rootName
    if not os.path.exists(root):
        os.makedirs(root)

    DIR00 = root + '/Original/'
    DIR00BW = root + '/OriginalBW/'
    DIR0 = root + '/Augmented/'
    DIR0BW = root + '/AugmentedBW/'
    DIR1 = root + '/TruthMask/'
    DIR2 = root + '/TruthJoint/'
    DIR3 = root + '/DetectedMask/'
    DIR4 = root + '/DetectedJoint/'

    if isDetected:
        if not os.path.exists(DIR3):
            os.makedirs(DIR3)
        if not os.path.exists(DIR3 + cla):
            os.makedirs(DIR3 + cla)
        if not os.path.exists(DIR4):
            os.makedirs(DIR4)
        if not os.path.exists(DIR4 + cla):
            os.makedirs(DIR4 + cla)

        cv2.imwrite(DIR3 + cla + '/' + imgName, mask)
        cv2.imwrite(DIR4 + cla + '/' + imgName, joint)
    else:
        if not os.path.exists(DIR00):
            os.makedirs(DIR00)
        if not os.path.exists(DIR00 + cla):
            os.makedirs(DIR00 + cla)
        if not os.path.exists(DIR00BW):
            os.makedirs(DIR00BW)
        if not os.path.exists(DIR00BW + cla):
            os.makedirs(DIR00BW + cla)
        if not os.path.exists(DIR0):
            os.makedirs(DIR0)
        if not os.path.exists(DIR0 + cla):
            os.makedirs(DIR0 + cla)
        if not os.path.exists(DIR0BW):
            os.makedirs(DIR0BW)
        if not os.path.exists(DIR0BW + cla):
            os.makedirs(DIR0BW + cla)
        if not os.path.exists(DIR1):
            os.makedirs(DIR1)
        if not os.path.exists(DIR1 + cla):
            os.makedirs(DIR1 + cla)
        if not os.path.exists(DIR2):
            os.makedirs(DIR2)
        if not os.path.exists(DIR2 + cla):
            os.makedirs(DIR2 + cla)

        cv2.imwrite(DIR00 + cla + '/' + imgName, original)
        cv2.imwrite(DIR00BW + cla + '/' + imgName, gray_imageOrig)
        cv2.imwrite(DIR0 + cla + '/' + imgName, img)
        cv2.imwrite(DIR0BW + cla + '/' + imgName, gray_image)
        cv2.imwrite(DIR1 + cla + '/' + imgName, mask)
        cv2.imwrite(DIR2 + cla + '/' + imgName, joint)


for root, dirs, files in sorted(os.walk(ROOT_DIR)):
    for imgName in files:
        if not imgName.startswith('.'):
            ipath = (os.path.join(root, imgName))
            cla = os.path.basename(root)
            k = root.rfind("/")
            root2 = root[:k]
            #print(root2, cla, imgName)
            if GENERATE_OR_SEGMENT == 0:
                generateER(root2, cla, imgName)
            else:
                segmentER(root2, cla, imgName)

print(time.ctime())

import numpy as np
import cv2
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from skimage.measure import LineModelND, ransac
from skimage.color import rgb2hsv
from scipy.spatial import distance
from sklearn.cluster import MiniBatchKMeans
import os
import datetime
import itertools
import time

print(time.ctime())

def medianColor(frame, img):
	(x, y, h, w) = frame
	medRow = np.median(img[y + (h / 8):y + (h - (h / 8)),
						   x + (w / 8):x + (w - (w / 8))], axis=0);
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


def quality(groundTruth, mask, earMask, showMap=False):
	tpMask = 0 + (np.multiply(np.multiply(groundTruth, mask), earMask))
	tnMask = 0 + (np.multiply(np.multiply((1 - groundTruth), (1 - mask)), earMask))
	fpMask = 0 + (np.multiply(np.multiply((1 - groundTruth), mask), earMask))
	fnMask = 0 + (np.multiply(np.multiply(groundTruth, (1 - mask)), earMask))
	IoU = np.sum(0.0+tpMask) / (np.sum(0.0+fpMask) + np.sum(0.0+groundTruth));
	qualityMap = np.multiply(np.dstack((tpMask,tpMask,tpMask)), [255,0,0]) + np.multiply(np.dstack((tnMask,tnMask,tnMask)), [0,255,0]) + np.multiply(np.dstack((fpMask,fpMask,fpMask)), [0,0,255]) + np.multiply(np.dstack((fnMask,fnMask,fnMask)), [255,255,0]);
	if(showMap):
		plt.imshow(qualityMap)
		plt.show()

	tp = np.sum(tpMask)
	tn = np.sum(tnMask)
	fp = np.sum(fpMask)
	fn = np.sum(fnMask)
	return (tp, tn, fp, fn, qualityMap, IoU)


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

# img='9.jpg';
# earringMask = 0 + (cv2.imread('data/Mask/'+img, 0) > 240);
# eMask = 0 + (cv2.imread('data/MaskEar/'+img, 0) > 240);
# mask = segment('data/'+img, 28 * 5, 22 * 5, grabcutIter = 10, ransacThreashold = 30, preGamma = 0.5, postGamma = 1.2, showGraph = False, useFullImageRes = True)
#
# (tp,tn,fp,fn) = quality(earringMask,mask,eMask,showMap = True);
#
# print "precision: ", float(tp)/(tp+fp);
# print "recall: ", float(tp)/(tp+fn);


def runTest(dbPath, numberOfiter=1, invert=False):
	db = [name for name in os.listdir(dbPath) if name.endswith(".png")]
	if not os.path.exists(dbPath + "/Results"):
		os.makedirs(dbPath + "/Results")
	f = open(dbPath + "/Results/results.csv", 'w')
	f.write('"ImageName","AvgPrecision","avgRecall","bestPrecision","bestRecall","worstPrecision","worstRacall","TP","FP","TN","FN","IoU","executionTime(ms)","width","height","Params"\n')

	for i in db:

		#for param in itertools.product([1, 5, 10], [10, 20, 30], [0.2, 0.5], [1.1, 1.2]): #([1, 2, 5, 10], [10, 15, 20, 25, 30], [0.2, 0.5, 0.5], [1.1, 1.2, 1.3])
		param = [2, 15, 0.3, 1.1]
		avgPrec = 0
		avgRecall = 0

		bestPrec = 0
		bestRecall = 0

		worstPrec = 2
		worstRecall = 2
		for j in range(numberOfiter):
			img = i
			print("Testing image: " + img + ' iter: ' + str(j) + ' params: ' + str(param))
			if invert:
				earringMask = 0 + (cv2.imread(dbPath + '/Mask/' + img, 0) < 15)
			else:
				earringMask = 0 + (cv2.imread(dbPath + '/Mask/' + img, 0) > 240)

			if invert:
				eMask = 0 + (cv2.imread(dbPath + '/MaskEar/' + img, 0) < 15)
			else:
				eMask = 0 + (cv2.imread(dbPath + '/MaskEar/' + img, 0) > 240)
			image = cv2.imread(dbPath + '/' + img)
			height, width = image.shape[:2]
			a = datetime.datetime.now()
			try:
				mask = segment(image, 28 * 5, 22 * 5, grabcutIter=param[0], ransacThreashold=param[1], preGamma=param[2], postGamma=param[3], showGraph=False, useFullImageRes=True)
			except:
				continue;

			b = datetime.datetime.now()
			exTime = b - a;

			(tp, tn, fp, fn, map, IoU) = quality(
				earringMask, mask, eMask, showMap=False)
			if((tp + fp)>0):
				prec = float(tp) / (tp + fp)
			else:
				prec = 0;

			if ((tp + fn) > 0):
				recall = float(tp) / (tp + fn)
			else:
				recall = 0;
			if(bestPrec < prec):
				bestPrec = prec

			if(bestRecall < recall):
				bestRecall = recall

			if(worstPrec > prec):
				worstPrec = prec

			if(worstRecall > recall):
				worstRecall = recall

			imgName = img.split('-');
			className = imgName[0];
			sampleName = imgName[1];
				
			DIR0 = dbPath+'/Results/Map/' + className
			DIR1 = dbPath+'/Results/Mask/' + className
			DIR2 = dbPath+'/Results/Input/' + className
			DIR3 = dbPath+'/Results/Truth/' + className
			DIR4 = dbPath+'/Results/Joint/' + className
			DIR5 = dbPath+'/Results/JointTruth/' + className
			if not os.path.exists(DIR0):
				os.makedirs(DIR0)
			if not os.path.exists(DIR1):
				os.makedirs(DIR1)
			if not os.path.exists(DIR2):
				os.makedirs(DIR2)
			if not os.path.exists(DIR3):
				os.makedirs(DIR3)
			if not os.path.exists(DIR4):
				os.makedirs(DIR4)
			if not os.path.exists(DIR5):
				os.makedirs(DIR5)
				
			loMask = mask == 0
			mask[loMask] = 255
			hiMask = mask < 255
			mask[hiMask] = 0
			
			loMask = earringMask == 0
			earringMask[loMask] = 255
			hiMask = earringMask < 255
			earringMask[hiMask] = 0
			
			#print(image.shape, mask.shape)
			joint = np.dstack( ( image, mask ) )
			jointTruth = np.dstack( ( image, earringMask ) )
				
			avgPrec += prec
			avgRecall += recall
			cv2.imwrite(DIR0 + '/' + sampleName, map)
			cv2.imwrite(DIR1 + '/' + sampleName, mask)
			cv2.imwrite(DIR2 + '/' + sampleName, image)
			cv2.imwrite(DIR3 + '/' + sampleName, earringMask)
			cv2.imwrite(DIR4 + '/' + sampleName, joint)
			cv2.imwrite(DIR5 + '/' + sampleName, jointTruth)
			f.write(img + "," + str(avgPrec / numberOfiter) + "," + str(avgRecall / numberOfiter) + "," +
					str(bestPrec) + "," + str(bestRecall) + "," + str(worstPrec) + "," + str(worstRecall) + "," + str(tp) + "," + str(fp) + "," + str(tn) + "," + str(fn) + "," + str(IoU)+ "," + str(int(exTime.total_seconds() * 1000)) + "," + str(width) + "," + str(height) + "," + str(param) + "\n")

	f.close()


runTest("../data/uerc", numberOfiter=1, invert=True)

print(time.ctime())
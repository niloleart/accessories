import numpy as np
import cv2

dbPath = '../data/uerc'
COLOR = True

def loadImg(imgName):
	mask = cv2.imread(dbPath + '/Mask/' + imgName, 0)
	if COLOR:
		img = cv2.imread(dbPath + '/' + imgName)
		joint = np.dstack( ( img, mask ) )
	else:
		img = cv2.imread(dbPath + '/' + imgName, 0)
		joint = np.array([mask, mask, img])
		joint = np.transpose(joint, (1, 2, 0))
		
	cv2.imwrite('abakus.png', joint)
	w, h, d = joint.shape
	print(w, h, d)
	img = cv2.imread('abakus.png', -1)
	w, h, d = img.shape
	print(w, h, d)

loadImg('0001-01.png')

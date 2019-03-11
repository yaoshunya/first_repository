from PIL import Image
import os
import cv2
import numpy as np
import re
import pdb
import glob
"""
os.chdir('../data/train/images')
dir = os.getcwd()
files = os.listdir(dir)

len_images = len(files) #フォルダ内の画像の数
os.chdir('../../../code')
"""

os.chdir('../data/train')
files = glob.glob("images/*")

images = []
t=0
for i in files:
	img = cv2.imread(i)
	#images.appned(img)
	print(i)
	images.append(img)
	t = t+1
	if t == 100:
		break
	pdb.set_trace()
cv2.imshow('image_sample',images[0])
cv2.waitKey(0)
#pdb.set_trace()

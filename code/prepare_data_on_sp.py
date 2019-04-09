from PIL import Image
import os
import cv2
import numpy as np
import re
import pdb
import glob
import pickle
import argparse
import lie_learn.spaces.S2 as S2
from torchvision import datasets
import matplotlib.pylab as plt

"""
os.chdir('../data/train/images')
dir = os.getcwd()
files = os.listdir(dir)

len_images = len(files) #フォルダ内の画像の数
os.chdir('../../../code')
"""

NORTHPOLE_EPSILON = 1e-3
parser = argparse.ArgumentParser()
parser.add_argument("--bandwidth",
					help="the bandwidth of the S2 signal",
					type=int,
					default=500,
					required=False)
parser.add_argument("--chunk_size",
					help="size of image chunk with same rotation",
					type=int,
					default=5,
					required=False)
parser.add_argument("--noise",
					help="the rotational noise applied on the sphere",
					type=float,
					default=1.0,
					required=False)
args = parser.parse_args()



def rand_rotation_matrix(deflection=1.0,randnums=None):
	if randnums is None:
		randnums = np.random.uniform(size=(3,))
	#theta,phi,z=randnums
	#theta,phi,z=(2.83,4.98,0.90)
	#theta = theta * 2.0*deflection*np.pi
	#phi = phi * 2.0*np.pi
	#z = z * 2.0*deflection

	#theta,phi,z=(2.83,4.98,0.90)

	#theta,phi,z=(np.pi,2*np.pi/2,0.9)
	theta,phi,z=(np.pi*0.68,2*np.pi/2.0,0.9)
	#theta,phi,z=(3.0,4.7,0)

	r=np.sqrt(z)
	V=(
		np.sin(phi)*r,
		np.cos(phi)*r,
		np.sqrt(2.0-z)
	)

	st=np.sin(theta)
	ct=np.cos(theta)

	R=np.array(((ct,st,0),(-st,ct,0),(0,0,1)))

	M=(np.outer(V,V)-np.eye(3)).dot(R)

	with open('sample.txt',mode='w') as f:
		f.write(str(theta))
		f.write(str(phi))
		f.write(str(z))
		f.write('\n')

	#pdb.set_trace()

	return M

def rotate_grid(rot,grid):
	x,y,z=grid
	xyz=np.array((x,y,z))
	x_r,y_r,z_r=np.einsum('ij,jab->iab',rot,xyz)
	return x_r,y_r,z_r

def get_projection_grid(b, grid_type="Driscoll-Healy"):
	theta, phi = S2.meshgrid(b=b, grid_type=grid_type)
	x_ = np.sin(theta) * np.cos(phi)
	y_ = np.sin(theta) * np.sin(phi)
	z_ = np.cos(theta)
	#pdb.set_trace()
	return x_, y_, z_

def project_sphere_on_xy_plane(grid, projection_origin):
	sx, sy, sz = projection_origin
	x, y, z = grid
	#x = np.append(x[np.newaxis],y[np.newaxis],axis=0)
	#x = np.append(x,z[np.newaxis],axis=0)
	#cv2.imshow('sample',x.T)
	#cv2.waitKey(0)
	#if(np.all(z < 0)):
	#	print("z<0")
	z = z.copy() + 1
	#x = x.copy() + 1
	#if(np.all(z < 0)):
	#	print("z_<0")
	#x = x.copy() + 1#

	t = -z / (z - sz)
	qx = t * (x - sx) + x
	qy = t * (y - sy) + y

	xmin = 1/2 * (-1 - sx) + -1
	ymin = 1/2 * (-1 - sy) + -1

	rx = (qx - xmin) / (2 * np.abs(xmin))
	ry = (qy - ymin) / (2 * np.abs(ymin))
	#rx = qx
	#ry = qy
	#pdb.set_trace()

	return rx, ry

def sample_within_bounds(signal, x, y, bounds):
	xmin, xmax, ymin, ymax = bounds
	idxs = (xmin <= x) & (x < xmax) & (ymin <= y) & (y < ymax)

	if len(signal.shape) > 2:
		sample = np.zeros((signal.shape[0], x.shape[0], x.shape[1]))
		sample[:, idxs] = signal[:, x[idxs], y[idxs]]
	else:
		sample = np.zeros((x.shape[0], x.shape[1]))
		sample[idxs] = signal[x[idxs], y[idxs]]
	return sample

def sample_bilinear(signal, rx, ry):
	signal_dim_x = signal.shape[1]
	signal_dim_y = signal.shape[2]
	rx *= signal_dim_x*1.3
	ry *= signal_dim_y*1.3


	ix = rx.astype(int)
	iy = ry.astype(int)

	ix0 = ix - 1
	iy0 = iy - 1
	ix1 = ix + 1
	iy1 = iy + 1

	bounds = (0, signal_dim_x, 0, signal_dim_y)

	signal_00 = sample_within_bounds(signal, ix0, iy0, bounds)
	signal_10 = sample_within_bounds(signal, ix1, iy0, bounds)
	signal_01 = sample_within_bounds(signal, ix0, iy1, bounds)
	signal_11 = sample_within_bounds(signal, ix1, iy1, bounds)

	fx1 = (ix1-rx) * signal_00 + (rx-ix0) * signal_10
	fx2 = (ix1-rx) * signal_01 + (rx-ix0) * signal_11
	#cv2.imshow('smaple',((iy1 - ry) * fx1 + (ry - iy0) * fx2)[0])
	#cv2.waitKey()
	#pdb.set_trace()

	return (iy1 - ry) * fx1 + (ry - iy0) * fx2

def project_2d_on_sphere(signal, grid , projection_origin=None):


	if projection_origin is None:
		projection_origin = (0, 0, 2 + NORTHPOLE_EPSILON)

	rx, ry = project_sphere_on_xy_plane(grid, projection_origin)

	sample = sample_bilinear(signal, rx, ry)
	#pdb.set_trace()

	sample *= (grid[2] <= 1).astype(np.float64)
	sample_min = sample.min(axis=(1, 2)).reshape(-1, 1, 1)
	sample_max = sample.max(axis=(1, 2)).reshape(-1, 1, 1)

	sample = (sample - sample_min) / (sample_max - sample_min)
	sample *= 255
	sample = sample.astype(np.uint8)

	return sample

def divide_color(image):

	#image_b = np.array([])
	#image_g = np.array([])
	#image_r = np.array([])

	for i in range(image.shape[0]):
		if i ==0 :
			image_b = image[i].T[0].T[np.newaxis]
			image_g = image[i].T[1].T[np.newaxis]
			image_r = image[i].T[2].T[np.newaxis]
		else:
			image_b = np.append(image_b,image[i].T[0].T[np.newaxis],axis=0)
			image_g = np.append(image_g,image[i].T[1].T[np.newaxis],axis=0)
			image_r = np.append(image_r,image[i].T[2].T[np.newaxis],axis=0)
		print(i)


	return image_b,image_g,image_r

def create_sphere(data,grid):

	signals = data.reshape(-1,data.shape[1],data.shape[2]).astype(np.float64)
	n_signals = signals.shape[0]
	projections = np.ndarray(
		(signals.shape[0],2*args.bandwidth,2*args.bandwidth),dtype=np.uint8
	)
	current = 0
	#pdb.set_trace()
	while current < n_signals:
		idxs = np.arange(current,min(n_signals,current+args.chunk_size))
		chunk = signals[idxs]
		projections[idxs] = project_2d_on_sphere(chunk,grid)
		current += args.chunk_size
		print(current)

	return projections


def main():


	os.chdir('../data/train')
	files = glob.glob("images/*")

	#images = []
	images = np.array([])
	t=0
	size = (1242,375)
	size_ = (500,500)
	for i in files:
		img = cv2.imread(i)

		#img = cv2.resize(img,size_)

		if(img.shape[1] != 1242):
			img = cv2.resize(img,size)
			#cv2.imshow('sample',img)
			#cv2.waitKey(0) & 0xFF
		if(t == 0):
			images = np.array(img[np.newaxis])
		else:
			images = np.append(images,img[np.newaxis],axis=0)

		print(i)
		#img = cv2.resize(img,(5000,5000))
		#images.append(img)
		t = t+1
		if t == 10:
			break
	#pdb.set_trace()
	#sample = []
	img_b,img_g,img_r = divide_color(images)

	grid = get_projection_grid(b=args.bandwidth)
	rot = rand_rotation_matrix(deflection=args.noise)
	rotated_grid = rotate_grid(rot,grid)
	#pdb.set_trace()

	img_b = create_sphere(img_b,rotated_grid)
	img_g = create_sphere(img_g,rotated_grid)
	img_r = create_sphere(img_r,rotated_grid)

	for i in range(images.shape[0]):
		print(i)
		image_sample = np.append(img_b[i][np.newaxis],img_g[i][np.newaxis],axis=0)
		image_sample = np.append(image_sample,img_r[i][np.newaxis],axis=0)
		#pdb.set_trace()
		cv2.imwrite('sample{0}.png'.format(i),image_sample.T)
		cv2.waitKey(0) & 0xFF
		image_sample = []
	#pdb.set_trace()

	#cv2.imshow('sample',image_sample.T)
	#cv2.imwrite('sample.png',image_sample.T)
	#cv2.waitKey(0)




	"""
	grid = get_projection_grid(b=args.bandwidth)

	for i in range(200):
		data = images[i].T[0]
		#pdb.set_trace()
		signals = data.reshape(,data.shape[0],data.shape[1]).astype(np.float64)
		n_signals = signals.shape[0]
		current = 0
		projections = np.array(
			(signals.shape[0],2*args.bandwidth,2*args.bandwidth),
			dtype = np.uint8)

		while current < n_signals:
			idxs = np.arange(current,min(n_signals,current+args.chunk_size))
			chunk = signals[idxs]
			pdb.set_trace()
			projections[idxs] = project_2d_on_sphere(chunk,grid)
			current += args.chunk_size
			print("\r{0}/{1}".format(current,n_signals),end="")
		print("")

	pdb.set_trace()
	"""



		#sample.append(project_2d_on_sphere(signals,0))

	#sample = np.array(sample)
	#sample = np.reshape(sample,(sample.shape[0],sample.shape[2],sample.shape[3])).T

	#sample = project_2d_on_sphere(signals,0)
	#cv2.imshow('image',sample)
	#cv2.imwrite('sample.png',sample)
	#cv2.waitKey(0)
	#pdb.set_trace()

if __name__ == '__main__':
	main()
"""
cv2.imshow('image_sample',images[0])
cv2.waitKey(0)
"""
#pdb.set_trace()

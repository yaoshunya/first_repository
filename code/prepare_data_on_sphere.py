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
					default=10000	,
					required=False)
args = parser.parse_args()



def rand_rotation_matrix(deflection=1.0,randnums=None):
	if randnum is None:
		randnums = np.random.uniform(size=(3,))
	theta,phi,z=randnums

	theta = theta * 2.0*deflection*np.pi
	phi = phi * 2.0*np.pi
	z = z * 2.0*deflection

	r=np,sqrt(z)
	V=(
		np.sin(phi)*r,
		np.cos(phi)*r,
		np.sqrt(2.0-z)
	)

	st=np.sin(theta)
	ct=np.cos(theta)

	R=np.array(((ct,st,0),(-st,ct,0),(0,0,1)))

	M=(np.outer(V,V)-np.eye(3)).dot(R)

	return M

def rotate_grid(rot,grid):
	x,y,z=gridxyz=np.array((x,y,z))
	x_r,y_r,z_r=np.einsum('ij,jab->iab',rot,xyz)
	return x_r,y_r,z_r

def get_projection_grid(b, grid_type="Driscoll-Healy"):
	theta, phi = S2.meshgrid(b=b, grid_type=grid_type)
	x_ = np.sin(theta) * np.cos(phi)
	y_ = np.sin(theta) * np.sin(phi)
	z_ = np.cos(theta)
	return x_, y_, z_

def project_sphere_on_xy_plane(grid, projection_origin):
	sx, sy, sz = projection_origin
	x, y, z = grid
	z = z.copy() + 1

	t = -z / (z - sz)
	qx = t * (x - sx) + x
	qy = t * (y - sy) + y

	xmin = 1/2 * (-1 - sx) + -1
	ymin = 1/2 * (-1 - sy) + -1

	rx = (qx - xmin) / (2 * np.abs(xmin))
	ry = (qy - ymin) / (2 * np.abs(ymin))

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

	return (iy1 - ry) * fx1 + (ry - iy0) * fx2

def project_2d_on_sphere(signal, grid = 0, projection_origin=None):

	grid = get_projection_grid(b=args.bandwidth)

	if projection_origin is None:
		projection_origin = (0, 0, 2 + NORTHPOLE_EPSILON)
	rx, ry = project_sphere_on_xy_plane(grid, projection_origin)
	sample = sample_bilinear(signal, rx, ry)

	sample *= (grid[2] <= 1).astype(np.float64)
	sample_min = sample.min(axis=(1, 2)).reshape(-1, 1, 1)
	sample_max = sample.max(axis=(1, 2)).reshape(-1, 1, 1)

	sample = (sample - sample_min) / (sample_max - sample_min)
	sample *= 255
	sample = sample.astype(np.uint8)

	return sample


def main():


	os.chdir('../data/train')
	files = glob.glob("images/*")

	images = []
	t=0
	for i in files:
		img = cv2.imread(i)
		#pdb.set_trace()
		#images.appned(img)
		print(i)
		img = cv2.resize(img,(5000,5000))
		images.append(img)
		t = t+1
		if t == 4:
			break
	sample = []
	"""
	for i in range(images[2].shape[2]):
		data.append(images[2].T[i])

	data = np.array(data)

	for i in range(images[2].shape[2]):
		signals.append(data[i]).reshape(-1.1242,375).astype(np.float64)
	"""
	for i in range(images[2].shape[2]):

		data = images[2].T[i]
		signals = data.reshape(-1,5000,5000).astype(np.float64)

		sample.append(project_2d_on_sphere(signals,0))

	sample = np.array(sample)
	sample = np.reshape(sample,(sample.shape[0],sample.shape[2],sample.shape[3])).T

	#sample = project_2d_on_sphere(signals,0)
	cv2.imshow('image',sample)
	#cv2.imwrite('sample.png',sample)
	cv2.waitKey(0)
	#pdb.set_trace()

if __name__ == '__main__':
	main()
"""
cv2.imshow('image_sample',images[0])
cv2.waitKey(0)
"""
#pdb.set_trace()

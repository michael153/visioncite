import os
import random
from PIL import Image
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import assets
import settings
from build_model_torch import CNN

def load_model(path):
	print("Loading model...")
	model = CNN()
	model.load_state_dict(torch.load(path))
	print(model)
	print()
	model.eval()
	return model

def load_data(fname):
	print("Loading data...")
	img = Image.open(os.path.join(assets.IMAGE_PATH, str(fname) + '.jpeg'))
	assert tuple(img.size) == settings.DESIRED_RESOLUTION
	arr = np.asarray(img)
	arr = np.moveaxis(arr, -1, 0)
	arr = np.array([arr])
	print()
	return torch.Tensor(arr)

def draw_mask(data):
	def get_colors(n):
	    color = [''.join([random.choice('0123456789ABCDEF') for j in range(6)])
	             for i in range(n)]
	    return [tuple(int(col[i:i+2], 16) for i in (0, 2 ,4)) for col in color]
	data = data.detach().numpy()
	data = data[0]
	stats = []
	for i in range(len(settings.LABELS)):
		stats.append((np.average(data[i,:,:]), np.std(data[i,:,:])))
	print(data.shape)
	colors = get_colors(len(settings.LABELS))
	imgarr = np.zeros((settings.DESIRED_RESOLUTION[1], settings.DESIRED_RESOLUTION[0], 3), dtype=np.uint8)
	for r in range(settings.DESIRED_RESOLUTION[1]):
		for c in range(settings.DESIRED_RESOLUTION[0]):
			label = np.argmax(data[:,r,c])
			# label = 0
			# for k in range(len(settings.LABELS)-1, -1, -1):
			# 	if data[k,r,c] > stats[k][0] + 1*stats[k][1]:
			# 		label = k
			# 		break
			imgarr[r, c] = colors[label]
	img = Image.fromarray(imgarr)
	img.show()


version = 1556507101
model = load_model(os.path.join(assets.MODEL_PATH, str(version) + '/', 'saved_model'))
arr = load_data("00001288")

output = model(arr)
print(output.shape)
print(output)
draw_mask(output)
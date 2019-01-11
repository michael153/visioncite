import os
import time
import json
import subprocess
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from skimage import filters
import numpy as np
from PIL import Image, ImageDraw
from xml.dom import minidom
from subprocess import DEVNULL, STDOUT
from skimage.filters import threshold_otsu
from skimage import io, color

import images
import assets
import pipeline


def generate_otsu(img_file_name):
	img_path = os.path.join(images.RAW_IMAGES_PATH, '{0}.jpg'.format(img_file_name))
	binary_path = os.path.join(images.RAW_IMAGES_PATH, 'binary/{0}.jpg'.format(img_file_name))
	image = io.imread(img_path)
	image = color.rgb2grey(image)
	thresh = threshold_otsu(image)
	binary = image > thresh
	plt.imsave(binary_path, np.array(binary), cmap=plt.cm.gray)

def generate_segments(img_file_name, output_segments=True, save_image=True, display_image=True):
	try:
		use_binary = True
		prima_tesseract_src = '/Users/michaelwan/Downloads/TesseractToPAGE'
		if use_binary:
			img_path = os.path.join(images.RAW_IMAGES_PATH, 'binary/{0}.jpg'.format(img_file_name))
		else:
			img_path = os.path.join(images.RAW_IMAGES_PATH, '{0}.jpg'.format(img_file_name))
		xml_path = os.path.join(assets.SEGMENTS_PATH, 'xml/{0}.xml'.format(img_file_name))
		segment_path = os.path.join(assets.SEGMENTS_PATH, '{0}.segment'.format(img_file_name))

		image = Image.open(img_path)
		if save_image:
			bounded = image.copy()

		cmd = "wine '{0}/bin/PRImA_Tesseract-1-4-144.exe' -inp-img '{1}' -out-xml '{2}' -rec-mode layout>>log.txt".format(
				prima_tesseract_src,
				img_path,
				xml_path
			   )
		if display:
			print(cmd)
		start = time.time()
		subprocess.call("{0}".format(cmd), shell=True, stderr=DEVNULL, stdout=DEVNULL)
		if display:
			print("Time for PRImA-Tesseract Call: {0:.2f}".format(time.time() - start))
		if display:
			fig,ax = plt.subplots(1)
			ax.imshow(image, cmap=plt.cm.gray)

		xmldoc = minidom.parse(xml_path)
		regions = [xmldoc.getElementsByTagName('SeparatorRegion'),
				   xmldoc.getElementsByTagName('TextRegion'),
				   xmldoc.getElementsByTagName('ImageRegion')]
		region_names = ['Separator', 'Text', 'Region']
		segments = []

		for r_id, region in enumerate(regions):
			for i, section in enumerate(region):
				r_type = region_names[r_id]
				coord = section.getElementsByTagName('Coords')[0]
				points = [(int(p[:p.find(',')]), int(p[p.find(',')+1:])) for p in coord.attributes['points'].value.split(' ')]
				segment = []
				segment.append("[{0}]".format(r_type))
				segment.append(str(len(points)))
				for p in points:
					segment.append("({0}, {1})".format(p[0], p[1]))
				if display:
					rect = patches.Polygon(list(points), linewidth=1,edgecolor='r',facecolor='none')
					ax.add_patch(rect)
				if save_image:
					draw = ImageDraw.Draw(bounded)
					draw.polygon(list(points), outline='red')
				segments.append(segment)
		if output_segments:
			with open(segment_path, 'w') as f:
				f.write('\n'.join(['\n'.join(s) for s in segments]))
		if save_image:
			bounded.save(os.path.join(images.BOUNDED_IMAGES_PATH, '{0}.jpg'.format(img_file_name)), "JPEG")
		return segments
	except Exception as e:
		print("[{0}] Segments [Failed!]: {1}".format(image, e))
		return []


# '''
# Module One: Generating Segmented Images
# '''

# saved_images = pipeline.get_images()
# display = False
# # img_file_name = '1545518044OQX039A3HQ'
# for image in saved_images:
# 	try:
# 		generate_otsu(image)
# 		start = time.time()
# 		generate_segments(image, output_segments=True, save_image=True, display_image=False)
# 		dur = time.time() - start
# 		print("[{0}] Segments calculated in: {1:.2f}s".format(image, dur))
# 	except Exception as e:
# 		print("[{0}] Preprocessing [Failed!]: {1}".format(image, e))

# # if display:
# # 	plt.show()


'''
Module Two: Generating Training Data
'''
file_name = '1547083109_export.json'
data = pipeline.aggregate_labels(file_name)
train_file = file_name[:file_name.rfind('.')] + '.train'
with open(os.path.join(assets.DATA_PATH, 'training/{0}'.format(train_file)), 'w') as f:
	json.dump(data, f, sort_keys=True, indent=4)
import time
import subprocess
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from skimage import filters
import numpy as np
from PIL import Image
from xml.dom import minidom
from subprocess import DEVNULL, STDOUT
from skimage.filters import threshold_otsu
from skimage import io, color

import images
import assets
import pipeline


def generate_otsu(img_file_name):
	img_path = images.RAW_IMAGES_PATH + '/' + img_file_name + '.jpg'
	binary_path = images.RAW_IMAGES_PATH + '/binary/' + img_file_name + '.jpg'
	image = io.imread(img_path)
	image = color.rgb2grey(image)
	thresh = threshold_otsu(image)
	binary = image > thresh
	plt.imsave(binary_path, np.array(binary), cmap=plt.cm.gray)

def generate_segments(img_file_name, output=True, display=True):
	# img_file_name = '1545517694SILDA9G5I3'
	prima_tesseract_src = '/Users/michaelwan/Downloads/TesseractToPAGE'
	img_path = images.RAW_IMAGES_PATH + '/' + img_file_name + '.jpg'
	xml_path = assets.SEGMENTS_PATH + '/xml/' + img_file_name + '.xml'
	segment_path = assets.SEGMENTS_PATH + '/' + img_file_name + '.segment'

	image = io.imread(img_path)
	image = color.rgb2grey(image)

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
			segments.append(segment)
	if output:
		with open(segment_path, 'w') as f:
			f.write('\n'.join(['\n'.join(s) for s in segments]))
	return segments


display = False

img_file_name = '1545518044OQX039A3HQ'

start = time.time()
generate_segments(img_file_name)
print("Segments calculated in: {0:.2f}s".format(time.time() - start))

if display:
	plt.show()
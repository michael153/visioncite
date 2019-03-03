import assets
import json
import random
from PIL import Image, ImageDraw
from matplotlib.pyplot import cm
import numpy as np

from build_model import scale_img
from preprocessing import xml_to_json

def get_colors(n):
    color = [''.join([random.choice('0123456789ABCDEF') for j in range(6)])
             for i in range(n)]
    return [tuple(int(col[i:i+2], 16) for i in (0, 2 ,4)) for col in color]

def draw_overlay(img, data):
	print(json.dumps(data, indent=4))

	num_types = sum([len(data['xml'][region_type].keys()) for region_type in data['xml']], 0)
	colors = get_colors(num_types)
	color_index = 0

	width = data['metadata']['width']
	height = data['metadata']['height']
	img = img.convert('RGBA')
	overlay = Image.new('RGBA', (width, height), (255, 255, 255, 0))

	for region_type in data['xml']:
	    for region in data['xml'][region_type]:
	        for shape in data['xml'][region_type][region]:
	            poly = [tuple(p) for p in shape]
	            draw = ImageDraw.Draw(overlay)
	            draw.polygon(poly, fill=tuple(list(colors[color_index]) + [200]), outline=colors[color_index])
	        color_index += 1

	img = Image.alpha_composite(img, overlay)
	img = img.convert("RGB")
	img.show()
	print(num_types)


CONST_IMG_HEIGHT = 1800
CONST_IMG_WIDTH  = 1200

scale =  lambda cur: lambda new: lambda p: (int(p[0]*new[0]/cur[0]), int(p[1]*new[1]/cur[1]))

file_name = '00000086'
xml_file_name = file_name + '.xml'
img_file_name = file_name + '.tif'

data = xml_to_json(assets.XML_PATH + "/" + xml_file_name)
img = Image.open(assets.IMAGE_PATH + "/" + img_file_name)

# scaled_data = xml_to_json(assets.XML_PATH + "/" + xml_file_name, scale(
# 	(data['metadata']['width'], data['metadata']['height']))
# 	((CONST_IMG_WIDTH, CONST_IMG_HEIGHT))
# )
# scaled_img = scaled_img(img_file_name, CONST_IMG_HEIGHT, CONST_IMG_WIDTH)

draw_overlay(img, data)
# draw_overlay(scaled_img, scaled_data)
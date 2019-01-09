import os
from PIL import Image

import images

new_size = (750, 900)

for filename in os.listdir(images.RAW_IMAGES_PATH):
	path = images.RAW_IMAGES_PATH + '/' + filename
	if '.jpg' in filename:
		small_path = images.RAW_IMAGES_PATH + '/smaller/' + filename
		im = Image.open(path)
		im.thumbnail(new_size, Image.ANTIALIAS)
		im.save(small_path, "JPEG")
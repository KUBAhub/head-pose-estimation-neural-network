import cv2
from PIL import Image
import numpy as np 
import os
import glob
import shutil
import csv
import StringIO
import re

# resized image (90, 82)
def crop_n_resize(image_path):
	img = Image.open(image_path).convert("L")

	
	img = img.crop( (50,0,img.size[0]-50, 260) )
	

	new_width  = 90
	new_height = new_width * img.size[1] / img.size[0]

	img = img.resize((new_width, new_height), Image.ANTIALIAS)

	return img


def extract_pan_tilt(imageName):

	matchObj = re.match( r'^[a-zA-Z0-9]*([-|+][0-9]*)([-|+][0-9]*).jpg', imageName, re.M|re.I)

	if matchObj:
	    tilt = matchObj.group(1)
	    pan  = matchObj.group(2)   
	else:
	   pan = 1000			# pan tilt not found in filename
	   tilt = 1000

	return int(pan), int(tilt)


def convert_to_1d_pixels(img):
	np_img = np.array(img)



	pixels = np.reshape(np_img, 90*82)

	return pixels




def process_image(path):
	
	filename = os.path.basename(path)
	pan,tilt = extract_pan_tilt(filename)
	resized = crop_n_resize(path)
	_1d_pixels = convert_to_1d_pixels(resized)
	
	if pan == 0 and tilt == 0:
		direction = 0				# Front
	elif pan == 0 and tilt > 0:
		direction = 1				# Right
	elif pan == 0 and tilt < 0:		
		direction = 2				# Left
	elif pan > 0 and tilt == 0:
		direction = 3				# Top
	elif pan < 0 and tilt == 0:
		direction = 4				# Bottom
	elif pan > 0 and tilt > 0:
		direction = 5				# Top Right
	elif pan > 0 and tilt < 0:
		direction = 6				# Top Left
	elif pan < 0 and tilt > 0:
		direction = 7				# Bottom Right
	else:
		direction = 8				# Bottom Left


	return _1d_pixels, pan, tilt, direction


def  main():

	_src_dir = "/home/awais/Desktop/AI/head_pose/dataset/images"

	out1 = csv.writer(open("classification_dataset.csv","w"), delimiter=',',quoting=csv.QUOTE_MINIMAL)
		
	row_id = 0

	out1.writerow(['row_id', 'image', 'direction'])		# classification

	errors = open('errors.txt','w');

	for jpgfile in glob.iglob(os.path.join( _src_dir , "*.jpg")):
		_1d_pixels, pan, tilt, direction = process_image(jpgfile)
		if pan == 1000 and tilt == 1000:
			errors.write(jpgfile+"\n\r")
			continue


		s = StringIO.StringIO()
		np.savetxt(s, _1d_pixels, fmt='%i',newline=' ',delimiter='')
		pixels_str = s.getvalue()

		out1.writerow([row_id, pixels_str, direction])		# classification

		row_id += 1

		if row_id % 50 == 0:
			print row_id, "images processed"

main()
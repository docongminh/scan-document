import os
import cv2
import glob
import tqdm
import argparse
from skimage.filters import threshold_local
import pytesseract

def check_exist(path):
	try:
		if not os.path.exists(path):
			os.mkdir(path)
	except Exception:
		raise "please check your folder again"
		pass


def extract_text_from_image(image, binary_mode = False, lang='vie'):
	# Convert the warped image to grayscale, then threshold it
	# to give it that 'black and white' paper effect
	if binary_mode:
		_input = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
		T = threshold_local(_input, 11, offset=10, method="gaussian")
		_input = (_input > T).astype("uint8") * 255
	else:
		_input = image

	config = '-l {lang}'.format(lang=lang)
	text = pytesseract.image_to_string(_input, config=config)
	lines = text.splitlines()
	text = '\n'.join(l.strip() for l in lines if l.strip())
	return _input, text


if __name__ == '__main__':

	parser = argparse.ArgumentParser()
	parser.add_argument('--input', type=str, default='./images')
	parser.add_argument('--use_binary', type=bool, default=True)
	parser.add_argument('--output', type=str, default='./output')
	parser.add_argument('--binary_output', type=str, default='./binary')

	FLAGS = parser.parse_args()

	allow_type = ['jpg', 'png', 'JPG', 'PNG', 'JPEG', 'jpeg']
	all_images = os.listdir(FLAGS.input)
	for image in tqdm.tqdm(all_images):
		try:
			endswith = image.split('.')[-1]
			if endswith in allow_type:
				name = image.split('.')[0]
				path_to_image = os.path.join(FLAGS.input, image)
				imread = cv2.imread(path_to_image)
				output_image, text = extract_text_from_image(image=imread, binary_mode=FLAGS.use_binary)
				#
				if FLAGS.use_binary:
					check_exist(FLAGS.binary_output)
					binary_output = '{}/{}.jpg'.format(FLAGS.binary_output, name)
					cv2.imwrite(binary_output, output_image)
				check_exist(FLAGS.output)
				output_file = '{}/{}.txt'.format(FLAGS.output, name)
				with open(output_file, 'w') as f:
					f.write(text)
			else:
				print("----> not allow type file: {} - type {}".format(image, endswith))
		except Exception as e:
			with open('logs.txt', 'w') as f:
				f.write(str(e))
			continue



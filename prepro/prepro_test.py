import os
import json
import argparse
import string
# non-standard dependencies:
import h5py
import numpy as np
from scipy.misc import imread, imresize

def main(params):

	imgs = json.load(open(params['input_json'], 'r'))["images"]

	# create output h5 file
	N = len(imgs)
	f = h5py.File(params['output_h5'], "w")
	dset = f.create_dataset("images", (N,3,256,256), dtype='uint8') # space for resized images
	splits = f.create_dataset("splits", (N,), dtype="uint32")
	imgids = f.create_dataset("imageids", (N,), dtype="uint32")
	for i,img in enumerate(imgs):
		imgids[i] = img["id"]		
		# load the image
		I = imread(os.path.join(params['images_root'], img['file_name']))
		try:
				Ir = imresize(I, (256,256))
		except:
				print 'failed resizing image %s - see http://git.io/vBIE0' % (img['file_path'],)
				raise
		# handle grayscale input images
		if len(Ir.shape) == 2:
			Ir = Ir[:,:,np.newaxis]
			Ir = np.concatenate((Ir,Ir,Ir), axis=2)
		# and swap order of axes from (256,256,3) to (3,256,256)
		Ir = Ir.transpose(2,0,1)
		# write to h5
		dset[i] = Ir
		if i % 1000 == 0:
			print 'processing %d/%d (%.2f%% done)' % (i, N, i*100.0/N)
		splits[i] = 4
	f.close()

if __name__ == "__main__":

	parser = argparse.ArgumentParser()

	# input json
	parser.add_argument('--input_json', default='/data/coco/coco_raw.json', help='input json file to process into hdf5')
	parser.add_argument('--output_h5', default='/data/coco/cocotalk_challenge.h5', help='output h5 file')
	
	# options
	parser.add_argument('--images_root', default='/home/jiasen/dataset/coco/', help='root location in which images are stored, to be prepended to file_path in input json')

	args = parser.parse_args()
	params = vars(args) # convert to ordinary dict
	print 'parsed input parameters:'
	print json.dumps(params, indent = 2)
	main(params)

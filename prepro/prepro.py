import os
import json
import argparse
import string
# non-standard dependencies:
import h5py
import numpy as np
from scipy.misc import imread, imresize

def build_vocab(imgs, params):
	count_thr = params['word_count_threshold']

	# count up the number of words
	counts = {}
	for img in imgs:
		for txt in img['sentences']:
			for w in txt['tokens']:
				counts[w] = counts.get(w, 0) + 1
	cw = sorted([(count,w) for w,count in counts.iteritems()], reverse=True)
	print 'top words and their counts:'
	print '\n'.join(map(str,cw[:20]))

	# print some stats
	total_words = sum(counts.itervalues())
	print 'total words:', total_words
	bad_words = [w for w,n in counts.iteritems() if n <= count_thr]
	vocab = [w for w,n in counts.iteritems() if n > count_thr]
	bad_count = sum(counts[w] for w in bad_words)
	print 'number of bad words: %d/%d = %.2f%%' % (len(bad_words), len(counts), len(bad_words)*100.0/len(counts))
	print 'number of words in vocab would be %d' % (len(vocab), )
	print 'number of UNKs: %d/%d = %.2f%%' % (bad_count, total_words, bad_count*100.0/total_words)

	# lets look at the distribution of lengths as well
	sent_lengths = {}
	for img in imgs:
		for txt in img['sentences']:
			nw = len(txt['tokens'])
			sent_lengths[nw] = sent_lengths.get(nw, 0) + 1
	max_len = max(sent_lengths.keys())
	print 'max length sentence in raw data: ', max_len
	print 'sentence length distribution (count, number of words):'
	sum_len = sum(sent_lengths.values())
	for i in xrange(max_len+1):
		print '%2d: %10d	 %f%%' % (i, sent_lengths.get(i,0), sent_lengths.get(i,0)*100.0/sum_len)

	# lets now produce the final annotations
	if bad_count > 0:
		# additional special UNK token we will use below to map infrequent words to
		print 'inserting the special UNK token'
		vocab.append('UNK')
	
	return vocab

def encode_captions(imgs, params, wtoi):
	""" 
	encode all captions into one large array, which will be 1-indexed.
	also produces label_start_ix and label_end_ix which store 1-indexed 
	and inclusive (Lua-style) pointers to the first and last caption for
	each image in the dataset.
	"""

	max_length = params['max_length']
	N = len(imgs)
	M = sum(len(img['sentences']) for img in imgs) # total number of captions

	label_arrays = []
	label_start_ix = np.zeros(N, dtype='uint32') # note: these will be one-indexed
	label_end_ix = np.zeros(N, dtype='uint32')
	label_length = np.zeros(M, dtype='uint32')
	caption_counter = 0
	counter = 1
	for i,img in enumerate(imgs):
		n = len(img['sentences'])
		assert n > 0, 'error: some image has no captions'

		Li = np.zeros((n, max_length), dtype='uint32')
		for j,s in enumerate(img['sentences']):
			label_length[caption_counter] = min(max_length, len(s["tokens"])) # record the length of this sequence
			caption_counter += 1
			for k,w in enumerate(s["tokens"]):
				if k < max_length:
					Li[j,k] = wtoi.get(w, wtoi["UNK"])

		# note: word indices are 1-indexed, and captions are padded with zeros
		label_arrays.append(Li)
		label_start_ix[i] = counter
		label_end_ix[i] = counter + n - 1
		
		counter += n
	
	L = np.concatenate(label_arrays, axis=0) # put all the labels together
	assert L.shape[0] == M, 'lengths don\'t match? that\'s weird'
	assert np.all(label_length > 0), 'error: some caption had no words?'

	print 'encoded captions to array of size ', `L.shape`
	return L, label_start_ix, label_end_ix, label_length

def main(params):

	imgs = json.load(open(params['input_json'], 'r'))["images"]

	# tokenization and preprocessing

	# create the vocab
	vocab = build_vocab(imgs, params)
	itow = {i+1:w for i,w in enumerate(vocab)} # a 1-indexed vocab translation table
	wtoi = {w:i+1 for i,w in enumerate(vocab)} # inverse table

	# encode captions in large arrays, ready to ship to hdf5 file
	L, label_start_ix, label_end_ix, label_length = encode_captions(imgs, params, wtoi)

	# create output h5 file
	N = len(imgs)
	f = h5py.File(params['output_h5'], "r+")
	f.create_dataset("labels", dtype='uint32', data=L)
	f.create_dataset("label_start_ix", dtype='uint32', data=label_start_ix)
	f.create_dataset("label_end_ix", dtype='uint32', data=label_end_ix)
	f.create_dataset("label_length", dtype='uint32', data=label_length)
	dset = f.create_dataset("images", (N,3,256,256), dtype='uint8') # space for resized images
	splits = f.create_dataset("splits", (N,), dtype="uint32")
	imgids = f.get("imageids")[:]
	for i,img in enumerate(imgs):
		if "cocoid" not in img:
			img["cocoid"] = img["imgid"]
		assert img["cocoid"] == imgids[i]
		# load the image
		I = imread(os.path.join(params['images_root'], img['file_path']))
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
		split = img["split"]
		if split == "train":
			splits[i] = 1
		elif split == "val":
			splits[i] = 2
		elif split == "restval":
			splits[i] = 3
		elif split == "test":
			splits[i] = 4
	f.close()
	print 'wrote ', params['output_h5']

	# create output json file
	out = {}
	out['itow'] = itow # encode the (1-indexed) vocab
	json.dump(out, open(params['output_json'], 'w'))
	print 'wrote ', params['output_json']

if __name__ == "__main__":

	parser = argparse.ArgumentParser()

	# input json
	parser.add_argument('--input_json', default='/data/coco/coco_raw.json', help='input json file to process into hdf5')
	parser.add_argument('--output_json', default='/data/coco/cocotalk_challenge.json', help='output json file')
	parser.add_argument('--output_h5', default='/data/coco/cocotalk_challenge.h5', help='output h5 file')
	
	# options
	parser.add_argument('--max_length', default=18, type=int, help='max length of a caption, in number of words. captions longer than this get clipped.')
	parser.add_argument('--images_root', default='/home/jiasen/dataset/coco/', help='root location in which images are stored, to be prepended to file_path in input json')
	parser.add_argument('--word_count_threshold', default=5, type=int, help='only words that occur more than this number of times will be put in vocab')

	args = parser.parse_args()
	params = vars(args) # convert to ordinary dict
	print 'parsed input parameters:'
	print json.dumps(params, indent = 2)
	main(params)

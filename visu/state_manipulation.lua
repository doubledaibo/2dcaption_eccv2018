require 'torch'
require 'nn'
require 'nngraph'
require 'misc.TestDataLoaderResNet'
require 'image'

local utils = require 'misc.utils'
require 'misc.LanguageModel'
require 'misc.LanguageModelCriterion'

local net_utils = require 'misc.net_utils'
require 'misc.optim_updates'
-------------------------------------------------------------------------------
-- Input arguments and options
-------------------------------------------------------------------------------
cmd = torch.CmdLine()
cmd:text()
cmd:text('Train an Image Captioning model')
cmd:text()
cmd:text('Options')


-- Data input settings

cmd:option('-input_h5','coco.h5','path to the h5file containing the preprocessed dataset')
cmd:option('-input_json','coco.json','path to the json file containing additional info and vocab')
cmd:option('-cnn_model','resnet-152.t7','path to CNN model file containing the weights, Caffe format. Note this MUST be a VGGNet-16 right now.')

cmd:option('-start_from', '', 'path to a model checkpoint to initialize model weights from. Empty = don\'t')

-- Model settings
cmd:option('-batch_size',20,'what is the batch size in number of images per batch? (there will be x seq_per_img sentences)')

--actuall batch size = gpu_num * batch_size

cmd:option('-split', 'val', '')
cmd:option('-x_start', 1, '')
cmd:option('-y_start', 1, '')
cmd:option('-x_end', 7, '')
cmd:option('-y_end', 7, '')
cmd:option('-out', 'out.json', '')

-- misc
cmd:option('-backend', 'cudnn', 'nn|cudnn')
cmd:option('-id', '1', 'an id identifying this run/job. used in cross-val and appended when writing progress files')
cmd:option('-seed', 123, 'random number generator seed to use')
cmd:option('-gpuid', 0, 'which gpu to use. -1 = use CPU')

cmd:text()

-------------------------------------------------------------------------------
-- Basic Torch initializations
-------------------------------------------------------------------------------
local opt = cmd:parse(arg)
--torch.manualSeed(opt.seed)
torch.setdefaulttensortype('torch.FloatTensor') -- for CPU

if opt.gpuid >= 0 then
	require 'cutorch'
	require 'cunn'
	if opt.backend == 'cudnn' then require 'cudnn' end
	cutorch.manualSeed(opt.seed)
	cutorch.setDevice(opt.gpuid + 1) -- note +1 because lua is 1-indexed
end

-------------------------------------------------------------------------------
-- Create the Data Loader instance
-------------------------------------------------------------------------------
local loader = DataLoader{h5_file = opt.input_h5, json_file = opt.input_json, 
		batch_size = opt.batch_size, seq_per_img = opt.seq_per_img}

-------------------------------------------------------------------------------
-- Initialize the networks
-------------------------------------------------------------------------------
-- create protos from scratch
-- intialize language model
opt.vocab_size = loader:getVocabSize()

local loaded_checkpoint
local protos = {} 
if opt.start_from ~= '' then -- just copy to gpu1 params
	print('load from ' .. opt.start_from)
	loaded_checkpoint = torch.load(opt.start_from)
	protos.lm = loaded_checkpoint.protos.lm:cuda()
	protos.cnn_conv_fix = loaded_checkpoint.protos.cnn_conv_fix:cuda()
	protos.cnn_conv = loaded_checkpoint.protos.cnn_conv:cuda()
else
	error("need a checkpoint")
end

protos.lm:createClones()
collectgarbage() 

local split = opt.split

protos.cnn_conv:evaluate()
protos.lm:evaluate()
protos.cnn_conv_fix:evaluate()

local n = 0
local vocab = loader:getVocab()

local nbatch = loader:getnBatch(split)

local imgId_cell = {}
local predictions = {}

opt.beam_size = 3

local vocab = loader:getVocab()

loader:init_rand(split)
loader:reset_iterator(split)
for n = 1, nbatch do
	if n % 10 == 0 then print(n .. " / " .. nbatch) end

	local data = loader:run(split)
	
	data.images = data.images:cuda()

	local feats_conv_fix = protos.cnn_conv_fix:forward(data.images)
	local feats_conv = protos.cnn_conv:forward(feats_conv_fix)
	
	local logprobs, seqs
	local seqs = protos.lm:sample_beam_sprange({feats_conv}, opt) 
	local sents = net_utils.decode_sequence(vocab, seqs)
	
	for k = 1, #sents do	
		local img_id = data.img_id[k]
		local entry
		if imgId_cell[img_id] == nil then
			imgId_cell[img_id] = 1
			entry = {image_id = img_id, caption = sents[k]}
			table.insert(predictions, entry)
		end
		if n == 1 then
			print(string.format('image %s: %s', entry.image_id, entry.caption))
		end 
	end
end

utils.write_json(opt.out, predictions)
						

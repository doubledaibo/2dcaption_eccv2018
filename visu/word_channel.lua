require 'torch'
require 'nn'
require 'nngraph'
require 'misc.DataLoaderResNet'

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
cmd:option('-usegen', 0, '')
cmd:option('-topk', 10, '')
cmd:option('-cid', 0, '')
cmd:option('-wid', 0, '')

cmd:option('-img_channel_size', 2048,'the encoding size of the image.')
cmd:option('-img_map_size', 7, '')
cmd:option('-img_kernel_size', 1, '')
cmd:option('-seq_per_img',5,'number of captions to sample for each image during training. Done for efficiency since CNN forward pass is expensive. E.g. coco has 5 sents/image')

-- Optimization: General
cmd:option('-grad_clip',0.1,'clip gradients at this value (note should be lower than usual 5 because we normalize grads by both batch and seq_length)')
cmd:option('-dropout', 0.5, 'strength of dropout in the Language Model RNN')

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
opt.seq_length = loader:getSeqLength()
opt.batch_size = opt.batch_size * opt.seq_per_img

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

opt.rnn_channel_size = protos.lm.rnn_channel_size
opt.rnn_map_size = protos.lm.rnn_map_size

protos.expanderConv = nn.FeatExpanderConv(opt.seq_per_img):cuda()
protos.averagepooling = nn.SpatialAveragePooling(opt.rnn_map_size, opt.rnn_map_size):cuda()

protos.lm:createClones()
collectgarbage() 

local split = opt.split

protos.cnn_conv:evaluate()
protos.lm:evaluate()
protos.cnn_conv_fix:evaluate()

local n = 0
local vocab = loader:getVocab()

local seq_per_img = opt.seq_per_img
if opt.usegen == 1 then
	seq_per_img = 1
end

local nbatch = loader:getnBatch(split)

local imgId_cell = {}

local activation_before = torch.zeros(opt.batch_size, opt.vocab_size, opt.rnn_channel_size)
local activation_after = torch.zeros(opt.batch_size, opt.vocab_size, opt.rnn_channel_size)
local activation_diff_sum = torch.zeros(opt.vocab_size, opt.rnn_channel_size)
local activation_diff_cnt = torch.zeros(opt.vocab_size)

loader:init_rand(split)
loader:reset_iterator(split)
for n = 1, nbatch do
	if n % 10 == 0 then print(n .. " / " .. nbatch) end

	local data = loader:run(split)
	
	data.images = data.images:cuda()
	data.labels = data.labels:cuda()

	local feats_conv_fix = protos.cnn_conv_fix:forward(data.images)
	local feats_conv = protos.cnn_conv:forward(feats_conv_fix)
	local expanded_feats_conv = protos.expanderConv:forward(feats_conv)	
	
	local logprobs, seqs, hs
	if opt.usegen == 1 then
		seqs = protos.lm:sample_beam({feats_conv}, {beam_size = 3})
		logprobs = protos.lm:forward({feats_conv, seqs})
	else
		logprobs = protos.lm:forward({expanded_feats_conv, data.labels})
		seqs = data.labels
	end

	activation_before:zero()
	activation_after:zero()

	local avg_h = {}
	for t = 1, opt.seq_length do
		if protos.lm.state[t] == nil then break end
		avg_h[t] = protos.averagepooling:forward(protos.lm.state[t]):view(-1, opt.rnn_channel_size):float()
	end
	for k = 1, seqs:size(2) do 
		local appeared = {}  --ignore words that occur multiple times
		local tmax = 0
		local curw, w
		for t = 1, opt.seq_length do
			curw = seqs[{t, k}]
			if curw == 0 then break end
			tmax = t 
			for j = t, opt.seq_length do 	
				w = seqs[{j, k}]
				if w > 0 then
					activation_before[{k, w}]:add(avg_h[t][k])
				else 
					break
				end
			end
			if appeared[curw] == nil then appeared[curw] = 0 end
			appeared[curw] = appeared[curw] + 1
			for j = 1, t - 1 do
				w = seqs[{j, k}]
				activation_after[{k, w}]:add(avg_h[t][k])
			end
		end
		for t = 1, tmax - 1 do 
			w = seqs[{t, k}]
			if appeared[w] == 1 then
				activation_diff_sum[w]:add(activation_before[k][w] / t - activation_after[k][w] / (tmax - t))
				activation_diff_cnt[w] = activation_diff_cnt[w] + 1
			end
		end 
	end
end

local activation_diff_avg = activation_diff_sum:clone()
for i = 1, opt.vocab_size do
	if activation_diff_cnt[i] ~= 0 then
		activation_diff_avg[i]:div(activation_diff_cnt[i])
	else
		activation_diff_avg[i] = 0
	end
end

N = opt.topk
if opt.cid ~= 0 then
	f = io.open('sort_word_by_channel_' .. opt.cid .. '.txt', 'w')
	i = opt.cid
	local top_avgs, top_idx = activation_diff_avg[{{}, {i, i}}]:topk(N, 1, true, true)
	for j = 1, N do
		f:write(top_idx[j][1] .. " " .. vocab[tostring(top_idx[j][1])] .. " " .. i .. " " .. top_avgs[j][1] .. " " .. activation_diff_cnt[top_idx[j][1]] .. "\n")
	end
	f.close()
elseif opt.wid ~= 0 then
	f = io.open('sort_channel_by_word_' .. opt.wid .. '.txt', 'w')
	i = opt.wid
	local top_avgs, top_idx = activation_diff_avg[i]:topk(N, true, true)
	for j = 1, N do
		f:write(i .. " " .. vocab[tostring(i)] .. " " .. top_idx[j] .. " " .. top_avgs[j] .. " " .. activation_diff_cnt[i] .. "\n")
	end
	f.close()
else
	f = io.open('sort_channel_by_word.txt', 'w')
	for i = 1, opt.vocab_size do
		local top_avgs, top_idx = activation_diff_avg[i]:topk(N, true, true)
		for j = 1, N do
			f:write(i .. " " .. vocab[tostring(i)] .. " " .. top_idx[j] .. " " .. top_avgs[j] .. " " .. activation_diff_cnt[i] .. "\n")
		end
	end
	f.close()	
	f = io.open('sort_word_by_channel.txt', 'w')
	for i = 1, opt.rnn_channel_size do
		local top_avgs, top_idx = activation_diff_avg[{{}, {i, i}}]:topk(N, 1, true, true)
		for j = 1, N do
			f:write(top_idx[j][1] .. " " .. vocab[tostring(top_idx[j][1])] .. " " .. i .. " " .. top_avgs[j][1] .. " " .. activation_diff_cnt[top_idx[j][1]] .. "\n")
		end
	end
	f.close()		
end

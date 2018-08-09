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
cmd:option('-checkpoint_path', 'save/', 'folder to save checkpoints into (empty = this folder)')
cmd:option('-startEpoch', 1, 'Max number of training epoch')

cmd:option('-dataset', 'eval_coco_val', '')

-- Model settings
cmd:option('-batch_size',20,'what is the batch size in number of images per batch? (there will be x seq_per_img sentences)')
cmd:option('-iter_size',1,'')

-- training setting
cmd:option('-nEpochs', 50, 'Max number of training epoch')
cmd:option('-finetune_cnn_after', 21, 'After what epoch do we start finetuning the CNN? (-1 = disable; never finetune, 0 = finetune from start)')

--actuall batch size = gpu_num * batch_size

cmd:option('-rnn_channel_size', 256, 'the channel size of hidden state, and the enlarged word embedding')
cmd:option('-rnn_map_size', 7, 'the feature map size of hidden state, and the enlarged word embedding')
cmd:option('-rnn_kernel_size', 3, 'the kernel size of convolution via hidden state')
cmd:option('-word_channel_size', 4, 'the feature channel size of the original word embedding')
cmd:option('-word_map_size', 15, 'the feature map size of the original word embedding')
cmd:option('-celltype', 'rnn', 'type of cell')

cmd:option('-img_channel_size', 2048,'the encoding size of the image.')
cmd:option('-img_map_size', 7, '')
cmd:option('-img_kernel_size', 1, '')
cmd:option('-seq_per_img',5,'number of captions to sample for each image during training. Done for efficiency since CNN forward pass is expensive. E.g. coco has 5 sents/image')

-- Optimization: General
cmd:option('-grad_clip',0.1,'clip gradients at this value (note should be lower than usual 5 because we normalize grads by both batch and seq_length)')
cmd:option('-dropout', 0.5, 'strength of dropout in the Language Model RNN')

-- Optimization: for the Language Model
cmd:option('-optim','adam','what update to use? rmsprop|sgd|sgdmom|adagrad|adam')
cmd:option('-learning_rate',4e-4,'learning rate')
cmd:option('-learning_rate_decay_start', 20, 'at what iteration to start decaying learning rate? (-1 = dont)')
cmd:option('-learning_rate_decay_every', 50, 'how many epoch the learning rate x 0.5')
cmd:option('-optim_alpha',0.8,'alpha for adagrad/rmsprop/momentum/adam')
cmd:option('-optim_beta',0.999,'beta used for adam')
cmd:option('-optim_epsilon',1e-8,'epsilon that goes into denominator for smoothing')

-- Optimization: for the CNN
cmd:option('-cnn_optim','adam','optimization to use for CNN')
cmd:option('-cnn_optim_alpha',0.8,'alpha for momentum of CNN')
cmd:option('-cnn_optim_beta',0.999,'alpha for momentum of CNN')
cmd:option('-cnn_learning_rate',1e-5,'learning rate for the CNN')
cmd:option('-cnn_weight_decay', 0, 'L2 weight decay just for the CNN')
cmd:option('-finetune_start_layer', 6, 'finetune start layer. [1-10]')

-- Evaluation/Checkpointing
cmd:option('-save_checkpoint_every', 2000, 'how often to save a model checkpoint?')
cmd:option('-language_eval', 1, 'Evaluate language as well (1 = yes, 0 = no)? BLEU/CIDEr/METEOR/ROUGE_L? requires coco-caption code from Github.')

-- misc
cmd:option('-backend', 'cudnn', 'nn|cudnn')
cmd:option('-id', '1', 'an id identifying this run/job. used in cross-val and appended when writing progress files')
cmd:option('-seed', 123, 'random number generator seed to use')
cmd:option('-gpuid', 0, 'which gpu to use. -1 = use CPU')

cmd:option('-scheduled_sampling_start', -1, 'at what iteration to start decay gt probability')
cmd:option('-scheduled_sampling_increase_every', 5, 'every how many iterations thereafter to gt probability')
cmd:option('-scheduled_sampling_increase_prob', 0.05, 'How much to update the prob')
cmd:option('-scheduled_sampling_max_prob', 0.25, 'Maximum scheduled sampling prob.')

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

local loaded_checkpoint
local protos = {} 
if opt.start_from ~= '' then -- just copy to gpu1 params
	print('load from ' .. opt.start_from)
	loaded_checkpoint = torch.load(opt.start_from)
	protos.lm = loaded_checkpoint.protos.lm:cuda()
	protos.cnn_conv_fix = loaded_checkpoint.protos.cnn_conv_fix:cuda()
	protos.cnn_conv = loaded_checkpoint.protos.cnn_conv:cuda()
else
	protos.lm = nn.LanguageModel(opt):cuda()
	local cnn_raw = torch.load(opt.cnn_model)
	-- 1~5
	protos.cnn_conv_fix = net_utils.build_residual_cnn_conv_fix(cnn_raw, {backend = cnn_backend, start_layer_num = opt.finetune_start_layer}):cuda()
	-- 6~8
	protos.cnn_conv = net_utils.build_residual_cnn_conv(cnn_raw, {backend = cnn_backend, start_layer_num = opt.finetune_start_layer}):cuda()
end

-- layer that expands features out so we can forward multiple sentences per image
protos.expanderConv = nn.FeatExpanderConv(opt.seq_per_img):cuda()
-- criterion for the language model
protos.crit = nn.LanguageModelCriterion():cuda()

params, grad_params = protos.lm:getParameters()
cnn1_params, cnn1_grad_params = protos.cnn_conv:getParameters()

print('total number of parameters in LM: ', params:nElement())
print('total number of parameters in CNN_conv: ', cnn1_params:nElement())

assert(params:nElement() == grad_params:nElement())
assert(cnn1_params:nElement() == cnn1_grad_params:nElement())

protos.thin_lm = protos.lm:clone()
protos.lm:shareThinClone(protos.thin_lm)

protos.lm:createClones()
collectgarbage() 

-------------------------------------------------------------------------------
-- Validation evaluation
-------------------------------------------------------------------------------
local function evaluate_split(split)

	print('=> evaluating ...')
	-- setting to the evaluation mode, use only the first gpu
	protos.cnn_conv:evaluate()
	protos.lm:evaluate()
	protos.cnn_conv_fix:evaluate()

	local n = 0
	local loss_sum = 0
	local predictions = {}
	local vocab = loader:getVocab()
	local imgId_cell = {}

	local nbatch = loader:getnBatch(split)
	loader:init_rand(split)
	loader:reset_iterator(split)
	local n = 0
	for i = 1, nbatch do
		local data = loader:run(split)
		-- convert the data to cuda
		data.images = data.images:cuda()
		data.labels = data.labels:cuda()

		-- forward the model to get loss
		local feats_conv_fix = protos.cnn_conv_fix:forward(data.images)
		local feats_conv = protos.cnn_conv:forward(feats_conv_fix)

		local expanded_feats_conv = protos.expanderConv:forward(feats_conv)
		local logprobs = protos.lm:forward({expanded_feats_conv, data.labels})

		local loss = protos.crit:forward({logprobs, data.labels})
		loss_sum = loss_sum + loss
		n = n + protos.crit.n 
		-- forward the model to also get generated samples for each image
		local seq = protos.lm:sample_beam({feats_conv}, {beam_size = 3})
		local sents = net_utils.decode_sequence(vocab, seq)

		for k=1,#sents do
			local img_id = data.img_id[k]
			local entry
			if imgId_cell[img_id] == nil then -- make sure there are one caption for each image.
				imgId_cell[img_id] = 1
				entry = {image_id = img_id, caption = sents[k]}
				table.insert(predictions, entry)
			end
			if i == 1 then -- print the first batch
				print(string.format('image %s: %s', entry.image_id, entry.caption))
			end
		end
	end
	local lang_stats
	if opt.language_eval == 1 then
		local sampleOpt = {beam_size = 3}		
		lang_stats = net_utils.language_eval(predictions, {id = opt.id, dataset = opt.dataset}, sampleOpt)
	end

	return loss_sum / n, predictions, lang_stats
end

-------------------------------------------------------------------------------
-- train function
-------------------------------------------------------------------------------
local function Train(epoch)

	protos.cnn_conv:training()
	protos.lm:training()
	protos.cnn_conv_fix:training()

	grad_params:zero()

	-- setting the gradient of the CNN network
	if epoch >= opt.finetune_cnn_after and opt.finetune_cnn_after ~= -1 then
		cnn1_grad_params:zero()
	end
	local loss = 0
	local n = 0
	for i = 1, opt.iter_size do
		local data = loader:run('train')
		data.images = data.images:cuda()
		data.labels = data.labels:cuda()

		local feats_conv_fix =	protos.cnn_conv_fix:forward(data.images)
		local feats_conv = protos.cnn_conv:forward(feats_conv_fix)
	
		local expanded_feats_conv = protos.expanderConv:forward(feats_conv)

		local log_prob = protos.lm:forward({expanded_feats_conv, data.labels, ss_prob})
		loss = loss + protos.crit:forward({log_prob, data.labels})
		n = n + protos.crit.n
		local d_logprobs = protos.crit:backward({})
		local dexpanded_conv = protos.lm:backward({}, d_logprobs)
		
		if epoch >= opt.finetune_cnn_after and opt.finetune_cnn_after ~= -1 then
			net_utils.setBNGradient0(protos.cnn_conv)
			dconv = protos.expanderConv:backward(feats_conv, dexpanded_conv)
			
			local dummy = protos.cnn_conv:backward(feats_conv_fix, dconv)
		end
	end
	if epoch >= opt.finetune_cnn_after and opt.finetune_cnn_after ~= -1 then
		-- apply L2 regularization
		if opt.cnn_weight_decay > 0 then
			cnn1_grad_params:add(opt.cnn_weight_decay, cnn1_params)
		end
		cnn1_grad_params:div(n)
		cnn1_grad_params:clamp(-opt.grad_clip, opt.grad_clip)
	end
	grad_params:div(n)
	grad_params:clamp(-opt.grad_clip, opt.grad_clip)
	
	if opt.optim == 'rmsprop' then
		rmsprop(params, grad_params, learning_rate, opt.optim_alpha, opt.optim_epsilon, optim_state)
	elseif opt.optim == 'adam' then
		adam(params, grad_params, learning_rate, opt.optim_alpha, opt.optim_beta, opt.optim_epsilon, optim_state)
	else
		error('bad option opt.optim')
	end
		
	if epoch >= opt.finetune_cnn_after and opt.finetune_cnn_after ~= -1 then
		if opt.cnn_optim == 'sgd' then
			sgd(cnn1_params, cnn1_grad_params, cnn1_learning_rate)
		elseif opt.cnn_optim == 'sgdm' then
			sgdm(cnn1_params, cnn1_grad_params, cnn_learning_rate, opt.cnn_optim_alpha, cnn1_optim_state)
		elseif opt.cnn_optim == 'adam' then
			adam(cnn1_params, cnn1_grad_params, cnn_learning_rate, opt.cnn_optim_alpha, opt.cnn_optim_beta, opt.optim_epsilon, cnn1_optim_state)
		else
			error('bad option for opt.cnn_optim')
		end
	end	

	return loss / n
end


paths.mkdir(opt.checkpoint_path)

-------------------------------------------------------------------------------
-- Main loop
-------------------------------------------------------------------------------
optim_state = {}
cnn1_optim_state = {}
learning_rate = opt.learning_rate
cnn_learning_rate = opt.cnn_learning_rate

local loss0
local loss_history = {}
local val_lang_stats_history = {}
local val_loss_history = {}
local checkpoint_path = path.join(opt.checkpoint_path, 'model_' .. opt.id)

ss_prob = 0.0

--evaluate_split('val')
epoch = opt.startEpoch - 1
nbatch = math.floor(loader:getnBatch('train') / opt.iter_size)  
iter = nbatch * epoch

while true do
	iter = iter + 1
	if iter % nbatch == 1 then
		loader:init_rand('train')
		loader:reset_iterator('train')
		epoch = epoch + 1
		if epoch > opt.nEpochs then break end
		-- doing the learning rate decay
		if epoch > opt.learning_rate_decay_start and opt.learning_rate_decay_start >= 0 then
			local frac = (epoch - opt.learning_rate_decay_start) / opt.learning_rate_decay_every
			local decay_factor = math.pow(0.5, frac)
			learning_rate = learning_rate * decay_factor -- set the decayed rate
		end
		if epoch > opt.scheduled_sampling_start and opt.scheduled_sampling_start >= 0 then
			local frac = (epoch - opt.scheduled_sampling_start) / opt.scheduled_sampling_increase_every
			frac = math.floor(frac)
			ss_prob = math.min(opt.scheduled_sampling_increase_prob * frac, opt.scheduled_sampling_max_prob)
	 	end
		print('=> Training epoch # ' .. epoch)
		print('lm_learning_rate: ' .. learning_rate 
			.. ' cnn_learning_rate: ' .. cnn_learning_rate)
		print ('ss_prob: ' .. ss_prob)
	end

	local train_loss = Train(epoch)
	if iter % 10 == 0 then
		print('iter: ' .. iter .. " / " .. nbatch .. " epoch: " .. epoch .. ", loss: " .. train_loss)
		collectgarbage()
	end	
	-- save the model.
	if iter % opt.save_checkpoint_every == 0 then
		local checkpoint = {}
		local save_protos = {}
		save_protos.cnn_conv = net_utils.deepCopy(protos.cnn_conv):float():clearState()
		save_protos.cnn_conv_fix = net_utils.deepCopy(protos.cnn_conv_fix):float():clearState()
		save_protos.lm = protos.thin_lm
		checkpoint.protos = save_protos
	 
		checkpoint.vocab = loader:getVocab()
		torch.save(checkpoint_path .. '_iter' .. iter .. '.t7', checkpoint)
		print('wrote checkpoint to ' .. checkpoint_path .. '_iter' .. iter .. '.t7')

		local val_loss, val_predictions, lang_stats = evaluate_split('val')
		print('val loss for # ' .. epoch .. ' : ' .. val_loss)
	end
end


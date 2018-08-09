require 'nn'
require 'nngraph'

local word_embedding = {}

function word_embedding.word_embedding(opt)
	local rc = opt.rnn_channel_size
	local rk = opt.rnn_kernel_size
	local rd = opt.rnn_stride_size
	local rp = (rk - 1) / 2
	local rmp = opt.rnn_map_size
	local wc = opt.word_channel_size
	local wmp = opt.word_map_size
	local dropout = opt.dropout
	
	local inputs = {}
	local outputs = {}

	table.insert(inputs, nn.Identity()())

	local mid_channel_size = 32
	if (wc <= 2) then mid_channel_size = 16 end 
	local kl = (wmp - rmp)/2+1
	local step1 = inputs[1]
	local step2 = nn.ReLU()(nn.SpatialConvolution(wc, mid_channel_size, kl, kl)(step1))  --mid_map_size x 11 x 11
	if (dropout>0) then step2 = nn.Dropout(dropout)(step2) end
	local step3 = nn.ReLU()(nn.SpatialConvolution(mid_channel_size, rc, kl, kl)(step2))
	if (dropout>0) then step3 = nn.Dropout(dropout)(step3) end

	table.insert(outputs, step3)

	return nn.gModule(inputs, outputs)

end

return word_embedding


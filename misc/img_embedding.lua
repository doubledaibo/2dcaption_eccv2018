require 'nn'
require 'nngraph'

local img_embedding = {}

function img_embedding.img_embedding(opt)
	
	local rc = opt.rnn_channel_size
	local rk = opt.rnn_kernel_size
	local rmp = opt.rnn_map_size
	local imp = opt.img_map_size
	local ic = opt.img_channel_size
	local ik = opt.img_kernel_size
	local dropout = opt.dropout
	
	local inputs = {}
	local outputs = {}

	table.insert(inputs, nn.Identity()()) -- image feature 

	local conv_feat = inputs[1]  -- batch_size*2048*7*7
	local conv_feat_embed 
	if imp >= rmp then
		conv_feat_embed = nn.ReLU()(nn.SpatialConvolution(ic, rc, ik, ik)(conv_feat))
	else
		conv_feat_embed = nn.ReLU()(nn.SpatialFullConvolution(ic, rc, ik, ik)(conv_feat))
	end
	if dropout > 0 then conv_feat_embed = nn.Dropout(dropout)(conv_feat_embed) end
	
	table.insert(outputs, conv_feat_embed)	
	
	return nn.gModule(inputs, outputs)

end

return img_embedding


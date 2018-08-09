require 'nn'
require 'nngraph'
local utils = require 'misc.utils'

local RNN2ds = {}

function RNN2ds.rnn2ds(opt)
	local rc = opt.rnn_channel_size
	local rk = opt.rnn_kernel_size
	local rd = opt.rnn_stride_size
	local rp = (rk-1)/2

	-- there will be 2*n+1 inputs
	local inputs = {}
	table.insert(inputs, nn.Identity()()) -- indices giving the sequence of symbols
	table.insert(inputs, nn.Identity()()) -- conv_feat
	table.insert(inputs, nn.Identity()()) -- prev_h[L]

        local outputs = {}
	local x = inputs[1]
        local v = inputs[2]
	local prev_h = inputs[3]

	local conv_h = nn.SpatialConvolution(rc, rc, rk, rk, rd, rd, rp, rp)(prev_h)
	local conv_x = nn.SpatialConvolution(rc, rc, rk, rk, rd, rd, rp, rp)(x)
	local conv_v = nn.SpatialConvolution(rc, rc, rk, rk, rd, rd, rp, rp)(v)
		
	local merge = nn.CAddTable()({conv_h, conv_x, conv_v})
	local next_h = nn.ReLU()(merge)

        table.insert(outputs, next_h)

	return nn.gModule(inputs, outputs)
end


return RNN2ds


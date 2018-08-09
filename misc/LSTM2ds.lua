require 'nn'
require 'nngraph'
local utils = require 'misc.utils'

local LSTM2ds = {}

function LSTM2ds.lstm2ds(opt)
	local rc = opt.rnn_channel_size
	local rk = opt.rnn_kernel_size
	local rd = opt.rnn_stride_size
	local rp = (rk - 1) / 2

	-- there will be 2*n+1 inputs
	local inputs = {}
	table.insert(inputs, nn.Identity()()) -- word's inputs
	table.insert(inputs, nn.Identity()()) -- img's inputs
	table.insert(inputs, nn.Identity()()) 
	table.insert(inputs, nn.Identity()())
 
        local outputs = {}

	local x = inputs[1]
	local v = inputs[2]
	local prev_h = inputs[3]
	local prev_c = inputs[4]
	
	local in_gate = nn.CAddTable()({
		nn.SpatialConvolution(rc, rc, rk, rk, rd, rd, rp, rp)(prev_h),
		nn.SpatialConvolution(rc, rc, rk, rk, rd, rd, rp, rp)(x),
		nn.SpatialConvolution(rc, rc, rk, rk, rd, rd, rp, rp)(v)
	})
	in_gate = nn.Sigmoid()(in_gate)
	local forget_gate = nn.CAddTable()({
		nn.SpatialConvolution(rc, rc, rk, rk, rd, rd, rp, rp)(prev_h),
		nn.SpatialConvolution(rc, rc, rk, rk, rd, rd, rp, rp)(x),
		nn.SpatialConvolution(rc, rc, rk, rk, rd, rd, rp, rp)(v)
	})
	forget_gate = nn.Sigmoid()(forget_gate)
	local out_gate = nn.CAddTable()({
		nn.SpatialConvolution(rc, rc, rk, rk, rd, rd, rp, rp)(prev_h), 
		nn.SpatialConvolution(rc, rc, rk, rk, rd, rd, rp, rp)(x),
		nn.SpatialConvolution(rc, rc, rk, rk, rd, rd, rp, rp)(v)
	})
	out_gate = nn.Sigmoid()(out_gate)	
	local in_transform = nn.CAddTable()({
		nn.SpatialConvolution(rc, rc, rk, rk, rd, rd, rp, rp)(x),
		nn.SpatialConvolution(rc, rc, rk, rk, rd, rd, rp, rp)(v),
		nn.SpatialConvolution(rc, rc, rk, rk, rd, rd, rp, rp)(prev_h)
	})
	in_transform = nn.ReLU()(in_transform)	
	
	local next_c = nn.CAddTable()({
		nn.CMulTable()({forget_gate, prev_c}),
		nn.CMulTable()({in_gate, in_transform})
	})
	local next_h = nn.CMulTable()({out_gate, nn.ReLU()(next_c)})	
		
        table.insert(outputs, next_h)
	table.insert(outputs, next_c)

	return nn.gModule(inputs, outputs)
end

return LSTM2ds


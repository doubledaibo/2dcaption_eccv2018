require 'nn'
require 'nngraph'
local utils = require 'misc.utils'

local GRU2ds = {}

function GRU2ds.gru2ds(opt)
	local rc = opt.rnn_channel_size
	local rk = opt.rnn_kernel_size
	local rd = opt.rnn_stride_size
	local rp = (rk - 1) / 2

	-- there will be 2*n+1 inputs
	local inputs = {}
	table.insert(inputs, nn.Identity()()) -- word's inputs
	table.insert(inputs, nn.Identity()()) -- img's inputs
	table.insert(inputs, nn.Identity()()) 
 
        local outputs = {}

	local x = inputs[1]
	local v = inputs[2]
	local prev_h = inputs[3]
	
	local merge_update = nn.CAddTable()({
		nn.SpatialConvolution(rc, rc, rk, rk, rd, rd, rp, rp)(prev_h),
		nn.SpatialConvolution(rc, rc, rk, rk, rd, rd, rp, rp)(x),
		nn.SpatialConvolution(rc, rc, rk, rk, rd, rd, rp, rp)(v)
	})
	local merge_reset = nn.CAddTable()({
		nn.SpatialConvolution(rc, rc, rk, rk, rd, rd, rp, rp)(prev_h), 
		nn.SpatialConvolution(rc, rc, rk, rk, rd, rd, rp, rp)(x),
		nn.SpatialConvolution(rc, rc, rk, rk, rd, rd, rp, rp)(v)
	})

	local update_gate = nn.Sigmoid()(merge_update)
		
	local reset_gate = nn.Sigmoid()(merge_reset)
		
		
	local temp_h = nn.CAddTable()({
		nn.SpatialConvolution(rc, rc, rk, rk, rd, rd, rp, rp)(x),
		nn.SpatialConvolution(rc, rc, rk, rk, rd, rd, rp, rp)(v),
		nn.CMulTable()({reset_gate, nn.SpatialConvolution(rc, rc, rk, rk, rd, rd, rp, rp)(prev_h)})
	})
	temp_h = nn.ReLU()(temp_h)	
		
	local leftc = nn.CMulTable()({temp_h, nn.AddConstant(1, false)(nn.MulConstant(-1, false)(update_gate))})
	local rightc = nn.CMulTable()({prev_h, update_gate})
	local next_h = nn.CAddTable()({leftc, rightc})
		
        table.insert(outputs, next_h)

	return nn.gModule(inputs, outputs)
end


return GRU2ds


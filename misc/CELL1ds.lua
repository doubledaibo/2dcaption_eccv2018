require 'nn'
require 'nngraph'

local CELL1ds = {}
function CELL1ds.lstm1ds(opt)
	local input_encoding_size = opt.input_encoding_size
	local rnn_size = opt.rnn_size
		
	-- there will be 2*n+1 inputs
	local inputs = {}
	table.insert(inputs, nn.Identity()()) 
	table.insert(inputs, nn.Identity()())
	table.insert(inputs, nn.Identity()())
	table.insert(inputs, nn.Identity()())

	local x = inputs[1]
	local v = inputs[2]
	local prev_h = inputs[3]
	local prev_c = inputs[4]

	local outputs = {}

	local combine = nn.JoinTable(1, 1)({x, v, prev_h})
	local combine2h = nn.Linear(input_encoding_size * 2 + rnn_size, 4 * rnn_size)(combine)
			
	local reshaped = nn.Reshape(4, rnn_size)(combine2h)
	local n1, n2, n3, n4 = nn.SplitTable(2)(reshaped):split(4)
	local in_gate = nn.Sigmoid()(n1)
	local forget_gate = nn.Sigmoid()(n2)
	local out_gate = nn.Sigmoid()(n3)
	-- decode the write inputs
	local in_transform = nn.Tanh()(n4)
	-- perform the LSTM update
	local next_c = nn.CAddTable()({
			nn.CMulTable()({forget_gate, prev_c}),
			nn.CMulTable()({in_gate, in_transform})
		})
		-- gated cells form the output
	local next_h = nn.CMulTable()({out_gate, nn.Tanh()(next_c)})
		
	table.insert(outputs, next_h)
	table.insert(outputs, next_c)

	return nn.gModule(inputs, outputs)
end

function CELL1ds.gru1ds(opt)
	local input_encoding_size = opt.input_encoding_size
	local rnn_size = opt.rnn_size
		
	-- there will be 2*n+1 inputs
	local inputs = {}
	table.insert(inputs, nn.Identity()()) 
	table.insert(inputs, nn.Identity()())
	table.insert(inputs, nn.Identity()())

	local x = inputs[1]
	local v = inputs[2]
	local prev_h = inputs[3]

	local outputs = {}

	local combine = nn.JoinTable(1, 1)({x, v, prev_h})
	local combine2h = nn.Linear(input_encoding_size * 2 + rnn_size, 2 * rnn_size)(combine)
			
	local reshaped = nn.Reshape(2, rnn_size)(combine2h)
	local n1, n2 = nn.SplitTable(2)(reshaped):split(2)
	local update_gate = nn.Sigmoid()(n1)
	local reset_gate = nn.Sigmoid()(n2)

	local temp_h = nn.CAddTable()({
		nn.Linear(input_encoding_size, rnn_size)(x),
		nn.Linear(input_encoding_size, rnn_size)(v),
		nn.CMulTable()({reset_gate, nn.Linear(rnn_size, rnn_size)(prev_h)})
	})
	temp_h = nn.Tanh()(temp_h)
	
	local leftc = nn.CMulTable()({temp_h, nn.AddConstant(1, false)(nn.MulConstant(-1, false)(update_gate))})
	local rightc = nn.CMulTable()({prev_h, update_gate})
	local next_h = nn.CAddTable()({leftc, rightc})

	table.insert(outputs, next_h)

	return nn.gModule(inputs, outputs)
end

function CELL1ds.rnn1ds(opt)
	local input_encoding_size = opt.input_encoding_size
	local rnn_size = opt.rnn_size
		
	-- there will be 2*n+1 inputs
	local inputs = {}
	table.insert(inputs, nn.Identity()()) 
	table.insert(inputs, nn.Identity()())
	table.insert(inputs, nn.Identity()())

	local x = inputs[1]
	local v = inputs[2]
	local prev_h = inputs[3]

	local outputs = {}

	local combine = nn.JoinTable(1, 1)({x, v, prev_h})
	local combine2h = nn.Linear(input_encoding_size * 2 + rnn_size, rnn_size)(combine)
	local next_h = nn.Tanh()(combine2h)

	table.insert(outputs, next_h)

	return nn.gModule(inputs, outputs)
end


return CELL1ds


require 'nn'
local crit, parent = torch.class('nn.LanguageModelCriterion', 'nn.Criterion')
function crit:__init()
	parent.__init(self)
end

function crit:updateOutput(inputs)
	local input = inputs[1]
	local seq = inputs[2]
	--local seq_len = inputs[3]

	local L,N,Mp1 = input:size(1), input:size(2), input:size(3)
	local D = seq:size(1)
	assert(D == L-1, 'input Tensor should be 1 larger in time')

	self.gradInput:resizeAs(input):zero()
	local loss = 0
	local n = 0
	for b=1,N do -- iterate over batches
		local first_time = true
		for t=1,L do -- iterate over sequence time (ignore t=1, dummy forward for the image)
			-- fetch the index of the next token in the sequence
			local target_index
			if t > D then -- we are out of bounds of the index sequence: pad with null tokens
				target_index = 0
			else
				target_index = seq[{t,b}]
			end
			-- the first time we see null token as next index, actually want the model to predict the END token
			if target_index == 0 and first_time then
				target_index = Mp1
				first_time = false
			end

			-- if there is a non-null next token, enforce loss!
			if target_index ~= 0 then
				-- accumulate loss
				loss = loss - input[{ t,b,target_index }] -- log(p)
				self.gradInput[{ t,b,target_index }] = -1
				n = n + 1
			end
		end
	end
	self.n = n
	self.output = loss
	
	return self.output
end

function crit:updateGradInput(inputs)
	return self.gradInput
end

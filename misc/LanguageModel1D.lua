require 'nn'
require 'misc.LookupTableMaskZero'
local utils = require 'misc.utils'
local net_utils = require 'misc.net_utils'
local CELL1ds = require 'misc.CELL1ds'

-------------------------------------------------------------------------------
-- Language Model core
-------------------------------------------------------------------------------

local layer, parent = torch.class('nn.LanguageModel', 'nn.Module')
function layer:__init(opt)
	parent.__init(self)

	-- options for core network
	self.vocab_size = opt.vocab_size
	self.input_encoding_size = opt.input_encoding_size
	self.rnn_size = opt.rnn_size
	self.fc_feat_size = opt.fc_feat_size
	self.dropout = opt.dropout
	self.seq_length = opt.seq_length
	self.celltype = opt.celltype
	
	if self.celltype == "rnn" then
		self.core = CELL1ds.rnn1ds(opt)
	elseif self.celltype == "gru" then
		self.core = CELL1ds.gru1ds(opt)
	elseif self.celltype == "lstm" then
		self.core = CELL1ds.lstm1ds(opt)
	end
	self.word_embedding = nn.LookupTableMaskZero(self.vocab_size + 1, self.input_encoding_size)
	self.img_embedding = nn.Sequential()
				:add(nn.Linear(self.fc_feat_size, self.input_encoding_size))
				:add(nn.ReLU())
				:add(nn.Dropout(self.dropout))
	self.logit = nn.Sequential()
			:add(nn.Dropout(self.dropout))
			:add(nn.Linear(self.rnn_size, self.vocab_size + 1))
			:add(nn.LogSoftMax())

	self.dv = torch.Tensor()
	
	self:_createInitState(1) -- will be lazily resized later during forward passes
end

function layer:_createInitState(batch_size)
	assert(batch_size ~= nil, 'batch size must be provided')
	if self.celltype == "lstm" then
		if not self.init_state then self.init_state = {} end -- lazy init
		for h = 1, 2 do
			-- note, the init state Must be zeros because we are using init_state to init grads in backward call too
			if self.init_state[h] then
				if self.init_state[h]:size(1) ~= batch_size then
					self.init_state[h]:resize(batch_size, self.rnn_size):zero() -- expand the memory
				end
			else
				self.init_state[h] = torch.zeros(batch_size, self.rnn_size)
			end
		end
		self.num_state = #self.init_state
	else
		if self.init_state then
			if self.init_state:size(1) ~= batch_size then
				self.init_state:resize(batch_size, self.rnn_size):zero()
			end
		else
			self.init_state = torch.zeros(batch_size, self.rnn_size)
		end
	end
end

function layer:createClones()
	-- construct the net clones
	print('constructing clones inside the LanguageModel')
	self.clones = {self.core}
	self.word_embeddings = {self.word_embedding}
	self.logits = {self.logit}
	for t = 2, self.seq_length + 1 do
		self.clones[t] = self.core:clone('weight', 'bias', 'gradWeight', 'gradBias')
		self.word_embeddings[t] = self.word_embedding:clone('weight', 'gradWeight')
		self.logits[t] = self.logit:clone('weight', 'bias', 'gradWeight', 'gradBias')
	end
end

function layer:shareThinClone(thin_copy)
	thin_copy.core:share(self.core, 'weight', 'bias')
	thin_copy.word_embedding:share(self.word_embedding, 'weight', 'bias')	
	thin_copy.img_embedding:share(self.img_embedding, 'weight', 'bias')
	thin_copy.logit:share(self.logit, 'weight', 'bias')
end

function layer:getModulesList()
	return {self.core, self.word_embedding, self.img_embedding, self.logit}
end

function layer:parameters()
	-- we only have two internal modules, return their params
	local params = {}
	local grad_params = {}
	
	local modules = self:getModulesList()
	for dummy, module in pairs(modules) do
		local p, g = module:parameters()
		for k, v in pairs(p) do table.insert(params, v) end
		for k, v in pairs(g) do table.insert(grad_params, v) end
	end

	return params, grad_params
end

function layer:training()
	for k,v in pairs(self.clones) do v:training() end
	for k,v in pairs(self.word_embeddings) do v:training() end
	for k,v in pairs(self.logits) do v:training() end
	self.img_embedding:training()
end

function layer:evaluate()
	for k,v in pairs(self.clones) do v:evaluate() end
	for k,v in pairs(self.word_embeddings) do v:evaluate() end
	for k,v in pairs(self.logits) do v:evaluate() end
	self.img_embedding:evaluate()
end

function layer:sample_beam(inputs, opt)
	local beam_size = utils.getopt(opt, 'beam_size', 3)

	local v = inputs[1]

	local batch_size = v:size(1)
	local function compare(a,b) return a.p > b.p end -- used downstream

	assert(beam_size <= self.vocab_size+1, 'lets assume this for now, otherwise this corner case causes a few headaches down the road. can be dealt with in future if needed')

	local v_embed = self.img_embedding:forward(v)

	local seq = torch.LongTensor(self.seq_length, batch_size):zero()
	local seqLogprobs = torch.FloatTensor(self.seq_length, batch_size):zero()
	local seqLogprobs_sum = torch.FloatTensor(batch_size):zero()

	-- lets process every image independently for now, for simplicity
	for k=1,batch_size do

		-- create initial states for all beams
		self:_createInitState(beam_size)
		local state = self.init_state

		-- we will write output predictions into tensor seq
		local beam_seq = torch.LongTensor(self.seq_length, beam_size):zero()
		local beam_seq_logprobs = torch.FloatTensor(self.seq_length, beam_size):zero()
		local beam_logprobs_sum = torch.zeros(beam_size) -- running sum of logprobs for each beam
		local logprobs -- logprobs predicted in last time step, shape (beam_size, vocab_size+1)
		local done_beams = {}
		
		--print (conv_feat[{ {k,k} }]:size())
		local v_embed_k = v_embed[{{k, k}}]:expand(beam_size, v_embed:size(2))
		
		--print (conv_feat_k:size())
		for t = 1, self.seq_length+1 do

			local xt, it, sampleLogprobs
			local new_state
			if t == 1 then
				-- feed in the start tokens
				local it = torch.LongTensor(beam_size):fill(self.vocab_size+1)
				xt = self.word_embedding:forward(it) -- NxK sized input (token embedding vectors)
			else	
				local logprobsf = logprobs:float() -- lets go to CPU for more efficiency in indexing operations
				ys, ix = torch.sort(logprobsf,2,true) -- sorted array of logprobs along each previous beam (last true = descending)
				local candidates = {}
				local cols = math.min(beam_size,ys:size(2))
				local rows = beam_size
				if t == 2 then rows = 1 end -- at first time step only the first beam is active
				for c=1,cols do -- for each column (word, essentially)
					for q=1,rows do -- for each beam expansion
						-- compute logprob of expanding beam q with word in (sorted) position c
						local local_logprob = ys[{ q,c }]
						local candidate_logprob = beam_logprobs_sum[q] + local_logprob
						table.insert(candidates, {c=ix[{ q,c }], q=q, p=candidate_logprob, r=local_logprob })
					end
				end
				table.sort(candidates, compare) -- find the best c,q pairs

				-- construct new beams
				if self.celltype == "lstm" then
					new_state = net_utils.clone_list(state)
				else
					new_state = state:clone():zero()
				end
				local beam_seq_prev, beam_seq_logprobs_prev
				if t > 2 then
					-- well need these as reference when we fork beams around
					beam_seq_prev = beam_seq[{ {1,t-2}, {} }]:clone()
					beam_seq_logprobs_prev = beam_seq_logprobs[{ {1,t-2}, {} }]:clone()
				end

				for vix=1,beam_size do
					local v = candidates[vix]
					-- fork beam index q into index vix
					if t > 2 then
						beam_seq[{ {1,t-2}, vix }] = beam_seq_prev[{ {}, v.q }]
						beam_seq_logprobs[{ {1,t-2}, vix }] = beam_seq_logprobs_prev[{ {}, v.q }]
					end
					if self.celltype == "lstm" then
						for state_ix = 1, self.num_state do
							new_state[state_ix][vix] = state[state_ix][v.q]
						end
					else
						new_state[vix] = state[v.q]
					end
					-- append new end terminal at the end of this beam
					beam_seq[{ t-1, vix }] = v.c -- c'th word is the continuation
					beam_seq_logprobs[{ t-1, vix }] = v.r -- the raw logprob here
					beam_logprobs_sum[vix] = v.p -- the new (sum) logprob along this beam

					if v.c == self.vocab_size + 1 or t == self.seq_length + 1 then
						-- END token special case here, or we reached the end.
						-- add the beam to a set of done beams
						if v.c == self.vocab_size + 1 then 
							beam_seq[{ t-1, vix }] = 0
						end
						table.insert(done_beams, {seq = beam_seq[{ {}, vix }]:clone(), 
							logps = beam_seq_logprobs[{ {}, vix }]:clone(),
							p = beam_logprobs_sum[vix]
						})
					end
				end
				
				-- encode as vectors
				it = beam_seq[t - 1]
				xt = self.word_embedding:forward(it)
			end

			if new_state then state = new_state end 
			local inputs
			if self.celltype == "lstm" then
				inputs = {xt, v_embed_k, unpack(state)}
			else
				inputs = {xt, v_embed_k, state}
			end
			local out = self.core:forward(inputs)
			if self.celltype == "lstm" then
				state = {}
				for i= 1, self.num_state do table.insert(state, out[i]) end
				logprobs = self.logit:forward(out[1])
			else
				state = out
				logprobs = self.logit:forward(out)
			end
		end

		table.sort(done_beams, compare)
		seq[{ {}, k }] = done_beams[1].seq -- the first beam has highest cumulative score
		seqLogprobs[{ {}, k }] = done_beams[1].logps
		seqLogprobs_sum[k] = done_beams[1].p
	end

	-- return the samples and their log likelihoods
	return seq, seqLogprobs_sum
end

function layer:updateOutput(input)
	local v = input[1]
	local seq = input[2]

	if self.clones == nil then self:createClones() end -- lazily create clones on first forward pass

	assert(seq:size(1) == self.seq_length)
	local batch_size = seq:size(2)

	self.output:resize(self.seq_length + 1, batch_size, self.vocab_size + 1):zero()
	
	self:_createInitState(batch_size)

	self.v = v
	self.v_embed = self.img_embedding:forward(v)

	self.state = {[0] = self.init_state}	
	self.inputs = {}
	self.word_embeddings_inputs = {}
	self.tmax = 0

	for t = 1, self.seq_length + 1 do

		local can_skip = false
		local xt
		if t == 1 then
			-- feed in the start tokens
			local it = torch.LongTensor(batch_size):fill(self.vocab_size + 1)
			self.word_embeddings_inputs[t] = it
			xt = self.word_embeddings[t]:forward(it)
		else
			-- feed in the rest of the sequence...
			local it = seq[t - 1]:clone()
			if torch.sum(it) == 0 then
				can_skip = true 
			else 
				self.word_embeddings_inputs[t] = it
				xt = self.word_embeddings[t]:forward(it)
			end
		end

		if not can_skip then
			if self.celltype == "lstm" then
				self.inputs[t] = {xt, self.v_embed, unpack(self.state[t - 1])}
			else
				self.inputs[t] = {xt, self.v_embed, self.state[t - 1]}
			end
			local out = self.clones[t]:forward(self.inputs[t])
			if self.celltype == "lstm" then
				self.state[t] = {}
				for i = 1, self.num_state do table.insert(self.state[t], out[i]) end
				self.output[t] = self.logits[t]:forward(out[1])
			else
				self.state[t] = out
				self.output[t] = self.logits[t]:forward(out)
			end
			self.tmax = t
		end
	end

	return self.output
end

function layer:updateGradInput(input, gradOutput)
	self.dv:resizeAs(self.v_embed):zero()
	local dstate = {[self.tmax] = self.init_state} -- this works when init_state is all zeros
	for t = self.tmax, 1, -1 do
		local dout = {}
		if self.celltype == "lstm" then
			for k = 1, self.num_state do table.insert(dout, dstate[t][k]) end
			dout[1] = dout[1] + self.logits[t]:backward(self.state[t][1], gradOutput[t])
		else
			dout = dstate[t] + self.logits[t]:backward(self.state[t], gradOutput[t])
		end
		local dinputs = self.clones[t]:backward(self.inputs[t], dout)
		local dxt = dinputs[1]
	
		self.dv:add(dinputs[2])
		self.word_embeddings[t]:backward(self.word_embeddings_inputs[t], dxt)
		if self.celltype == "lstm" then
			dstate[t - 1] = {}
			for k = 3, self.num_state + 2 do table.insert(dstate[t - 1], dinputs[k]) end
		else
			dstate[t - 1] = dinputs[3]
		end	
	end

	self.gradInput = self.img_embedding:backward(self.v, self.dv)
	return self.gradInput
end



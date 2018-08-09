local LookupTable2D, parent = torch.class('nn.LookupTable2D', 'nn.Module')

function LookupTable2D:__init(nIndex, outputC, outputK)
	parent.__init(self)
	self.weight = torch.Tensor(nIndex, outputC, outputK, outputK)
	self.gradWeight = torch.Tensor(nIndex, outputC, outputK, outputK)
	self.nIndex = nIndex
	self.outputC = outputC
	self.outputK = outputK
	self:reset()
end

function LookupTable2D:updateOutput(input)
	local n = input:size(1)
	self.output:resize(n, self.outputC, self.outputK, self.outputK):zero()
	for i = 1, n do
		if input[i] ~= 0 then
			self.output[i] = self.weight[input[i]]
		end
	end
	return self.output
end

function LookupTable2D:updateGradInput(input, gradOutput)
	return nil
end

function LookupTable2D:accGradParameters(input, gradOutput, scale)
	local n = input:size(1)
	for i = 1, n do
		if input[i] ~= 0 then	
			self.gradWeight[input[i]] = self.gradWeight[input[i]]:add(gradOutput[i]:mul(scale))
		end
	end	
end

function LookupTable2D:reset(stdv)
   if stdv then
      stdv = stdv * math.sqrt(3)
   else
      stdv = 1/math.sqrt(self.outputK*self.outputK*self.outputC)
   end
   if nn.oldSeed then
      self.weight:apply(function()
         return torch.uniform(-stdv, stdv)
      end)
      if self.bias then
         self.bias:apply(function()
         return torch.uniform(-stdv, stdv)
         end)
      end
   else
      self.weight:uniform(-stdv, stdv)
      if self.bias then
         self.bias:uniform(-stdv, stdv)
      end
   end
end

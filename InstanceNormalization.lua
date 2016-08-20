require 'nn'

_ = [[
   An implementation for https://arxiv.org/abs/1607.08022
]]

local InstanceNormalization, parent = torch.class('nn.InstanceNormalization', 'nn.Module')

function InstanceNormalization:__init(nOutput, eps, momentum, affine)
   self.running_mean = torch.zeros(nOutput)
   self.running_var = torch.ones(nOutput)

   if affine then 
      self.weight = torch.Tensor(nOutput)
      self.bias = torch.Tensor(nOutput)
      self.gradWeight = torch.Tensor(nOutput)
      self.gradBias = torch.Tensor(nOutput)
   end 

   self.eps = eps or 1e-5
   self.momentum = momentum or 0.1
   self.affine = affine

   self.nOutput = nOutput
   self.prev_batch_size = -1
end

function InstanceNormalization:updateOutput(input)
   self.output = self.output or input.new()
   assert(input:size(2) == self.nOutput)

   local batch_size = input:size(1)

   if batch_size ~= self.prev_batch_size or (self.bn and self:type() ~= self.bn:type())  then
      self.bn = nn.SpatialBatchNormalization(input:size(1)*input:size(2), self.eps, self.momentum, self.affine)
      self.bn:type(self:type())
      self.prev_batch_size = input:size(1)
   end

   -- Set params for BN
   self.bn.running_mean:copy(self.running_mean:repeatTensor(batch_size))
   self.bn.running_var:copy(self.running_var:repeatTensor(batch_size))
   if self.affine then
      self.bn.weight:copy(self.weight:repeatTensor(batch_size))
      self.bn.bias:copy(self.bias:repeatTensor(batch_size))
   end
   --

   local input_1obj = input:view(1,input:size(1)*input:size(2),input:size(3),input:size(4)) 
   self.output = self.bn:forward(input_1obj):viewAs(input)
  
   return self.output
end

function InstanceNormalization:updateGradInput(input, gradOutput)
   self.gradInput = self.gradInput or gradOutput.new()

   assert(self.bn)

   local input_1obj = input:view(1,input:size(1)*input:size(2),input:size(3),input:size(4)) 
   local gradOutput_1obj = gradOutput:view(1,input:size(1)*input:size(2),input:size(3),input:size(4)) 
  
   self.gradInput = self.bn:backward(input_1obj, gradOutput_1obj):viewAs(input)

   if self.affine then
      self.gradWeight:add(self.bn.gradWeight:view(input:size(1),self.nOutput):sum(1))
      self.gradBias:add(self.bn.gradBias:view(input:size(1),self.nOutput):sum(1))
   end
   return self.gradInput
end

function InstanceNormalization:clearState()
   self.output = self.output.new()
   self.gradInput = self.gradInput.new()
   
   self.bn:clearState()
end
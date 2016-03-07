loadcaffe_wrap = require 'src/loadcaffe_wrapper'

require 'src/SpatialCircularPadding'

----------------------------------------------------------
-- Shortcuts 
----------------------------------------------------------

function convc(in_,out_, k, s, m)
    print ('using convc')
    m = m or 1
    s = s or 1

    local pad = (k-1)/2*m

     if pad == 0 then
      return cudnn.SpatialConvolution(in_, out_, k, k, s, s, 0, 0)
    else

      local net = nn.Sequential()
      net:add(nn.SpatialCircularPadding(pad,pad,pad,pad))
      net:add(nn.SpatialConvolution(in_, out_, k, k, s, s, 0, 0))

      return net
    end
end

function conv(in_,out_, k, s, m)
    m = m or 1
    s = s or 1
    return nn.SpatialConvolution(in_, out_, k, k, s, s, (k-1)/2*m, (k-1)/2*m)
end

function bn(in_, m)
    return nn.SpatialBatchNormalization(in_,nil,m)
end

----------------------------------------------------------
--
----------------------------------------------------------


local GN, parent = torch.class('nn.GradNormalization', 'nn.Module')

function GN:updateOutput(input)
    self.output = input
   return self.output
end

function GN:updateGradInput(input, gradOutput)
   self.gradInput = self.gradInput or gradOutput:clone()
   if self.gradInput:nElement() ~=  gradOutput:nElement() then
     self.gradInput=  gradOutput:clone()
   end
   self.gradInput:copy(gradOutput)

   self.gradInput:div(torch.abs(self.gradInput):sum())
   return self.gradInput
end



---------------------------------------------------------
-- Helper function
---------------------------------------------------------

-- adds first dummy dimension
function torch.add_dummy(self)
  local sz = self:size()
  local new_sz = torch.Tensor(sz:size()+1)
  new_sz[1] = 1
  new_sz:narrow(1,2,sz:size()):copy(torch.Tensor{sz:totable()})
  return self:view(new_sz:long():storage())
end

function torch.FloatTensor:add_dummy()
  return torch.add_dummy(self)
end
function torch.DoubleTensor:add_dummy()
  return torch.add_dummy(self)
end

function torch.CudaTensor:add_dummy()
  return torch.add_dummy(self)
end


---------------------------------------------------------
-- DummyGradOutput
---------------------------------------------------------

-- Simpulates Identity operation with 0 gradOutput
local DummyGradOutput, parent = torch.class('nn.DummyGradOutput', 'nn.Module')

function DummyGradOutput:__init()
  parent.__init(self)
  self.gradInput = nil
end


function DummyGradOutput:updateOutput(input)
  self.output = input
  return self.output
end

function DummyGradOutput:updateGradInput(input, gradOutput)
  self.gradInput = self.gradInput or input.new():resizeAs(input):fill(0)
  if not input:isSameSizeAs(self.gradInput) then
    self.gradInput = self.gradInput:resizeAs(input):fill(0)
  end  
  return self.gradInput 
end


---------------------------------------------------------
-- Image processing
---------------------------------------------------------

-- Preprocess an image before passing it to a Caffe model.
-- We need to rescale from [0, 1] to [0, 255], convert from RGB to BGR,
-- and subtract the mean pixel.
function preprocess(img)
  local mean_pixel = torch.DoubleTensor({103.939, 116.779, 123.68})
  local perm = torch.LongTensor{3, 2, 1}
  img = img:index(1, perm):mul(255.0)
  mean_pixel = mean_pixel:view(3, 1, 1):expandAs(img)
  img:add(-1, mean_pixel)
  return img
end


-- Undo the above preprocessing.
function deprocess(img)
  local mean_pixel = torch.DoubleTensor({103.939, 116.779, 123.68})
  mean_pixel = mean_pixel:view(3, 1, 1):expandAs(img)
  img = img + mean_pixel
  local perm = torch.LongTensor{3, 2, 1}
  img = img:index(1, perm):div(255.0)
  return img
end

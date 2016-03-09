function deprocess(img)
  if img:dim() == 3 then
    local mean_pixel = torch.DoubleTensor({103.939, 116.779, 123.68})
    mean_pixel = mean_pixel:view(3, 1, 1):expandAs(img)
    img = img + mean_pixel
    local perm = torch.LongTensor{3, 2, 1}
    img = img:index(1, perm):div(256.0)
    img = torch.clamp(img,0,1)
    return img
  else
    local t = img:clone()
      for i=1,t:size(1) do
        t[i]:copy(deprocess(t[i]))
      end
     return t
  end
end

local GenNoise, parent = torch.class('nn.GenNoise', 'nn.Module')

function GenNoise:updateOutput(input)
    self.output = input

    if self.action == 'zero' then
        self.output:fill(0)
    elseif self.action == 'fix' then
        self.fix_value = self.fix_value or self.output.new():resize(self.output:size()):uniform()
        self.output = self.fix_value:clone()
    else           
        self.output:uniform()
    end
    
   return self.output
end

function GenNoise:updateGradInput(input, gradOutput)
   self.gradInput = gradOutput
   return self.gradInput
end

function zero_all_but(i, noise_modules)
    -- 
    noise_modules[1].action = 'zero'
    noise_modules[2].action = 'zero'
    noise_modules[3].action = 'zero'
    noise_modules[4].action = 'zero'
    noise_modules[5].action = 'zero'
    
    noise_modules[i].action = 'rand'
end

---------------------------------------------
-- SpatialCircularPadding
---------------------------------------------

local SpatialCircularPadding, parent = torch.class('nn.SpatialCircularPadding', 'nn.Module')

function SpatialCircularPadding:__init(pad_l, pad_r, pad_t, pad_b)
   parent.__init(self)
   self.pad_l = pad_l
   self.pad_r = pad_r or self.pad_l
   self.pad_t = pad_t or self.pad_l
   self.pad_b = pad_b or self.pad_l
end

function SpatialCircularPadding:updateOutput(input)
   if input:dim() == 4 then
      -- sizes
      local h = input:size(3) + self.pad_t + self.pad_b
      local w = input:size(4) + self.pad_l + self.pad_r
      if w < 1 or h < 1 then error('input is too small') end
      self.output:resize(input:size(1), input:size(2), h, w)
      self.output:zero()
      -- crop input if necessary
      local c_input = input
      if self.pad_t < 0 then c_input = c_input:narrow(3, 1 - self.pad_t, c_input:size(3) + self.pad_t) end
      if self.pad_b < 0 then c_input = c_input:narrow(3, 1, c_input:size(3) + self.pad_b) end
      if self.pad_l < 0 then c_input = c_input:narrow(4, 1 - self.pad_l, c_input:size(4) + self.pad_l) end
      if self.pad_r < 0 then c_input = c_input:narrow(4, 1, c_input:size(4) + self.pad_r) end
      -- crop outout if necessary
      local c_output = self.output
      if self.pad_t > 0 then c_output = c_output:narrow(3, 1 + self.pad_t, c_output:size(3) - self.pad_t) end
      if self.pad_b > 0 then c_output = c_output:narrow(3, 1, c_output:size(3) - self.pad_b) end
      if self.pad_l > 0 then c_output = c_output:narrow(4, 1 + self.pad_l, c_output:size(4) - self.pad_l) end
      if self.pad_r > 0 then c_output = c_output:narrow(4, 1, c_output:size(4) - self.pad_r) end
      -- copy input to output
      c_output:copy(c_input)

      self.output:narrow(3,1,1):copy(self.output:narrow(3,h - 1,1))
      self.output:narrow(3,h,1):copy(self.output:narrow(3,1 + 1,1))

      self.output:narrow(4,1,1):copy(self.output:narrow(4,w - 1,1))
      self.output:narrow(4,w,1):copy(self.output:narrow(4,1 + 1,1))
   else
      error('input must be 3 or 4-dimensional')
   end
   return self.output
end

function SpatialCircularPadding:updateGradInput(input, gradOutput)
   if input:dim() == 4 then
      self.gradInput = nil
      local cg_output = gradOutput
      if self.pad_t > 0 then cg_output = cg_output:narrow(3, 1 + self.pad_t, cg_output:size(3) - self.pad_t) end
      if self.pad_b > 0 then cg_output = cg_output:narrow(3, 1, cg_output:size(3) - self.pad_b) end
      if self.pad_l > 0 then cg_output = cg_output:narrow(4, 1 + self.pad_l, cg_output:size(4) - self.pad_l) end
      if self.pad_r > 0 then cg_output = cg_output:narrow(4, 1, cg_output:size(4) - self.pad_r) end
      -- copy gradOuput to gradInput
      self.gradInput = cg_output
   else
      error('input must be 4-dimensional')
   end
   return self.gradInput
end


function SpatialCircularPadding:__tostring__()
  return torch.type(self) ..
      string.format('(l=%d,r=%d,t=%d,b=%d)', self.pad_l, self.pad_r,
                    self.pad_t, self.pad_b)
end
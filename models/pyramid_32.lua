ratios = {32, 16, 8, 4, 2, 1}
cur = nil
local act = nn.LeakyReLU
conv_num = 8
cur_depth = nil

inplace = true

local GenNoise, parent = torch.class('nn.GenNoise', 'nn.Module')

function GenNoise:updateOutput(input)
    self.output = input
    self.output:narrow(2,1,3):uniform()
   return self.output
end

function GenNoise:updateGradInput(input, gradOutput)
   self.gradInput = gradOutput
   return self.gradInput
end

for i = 1,#ratios do
      
        seq = nn.Sequential()

        local tmp =  nn.SpatialAveragePooling(ratios[i],ratios[i],ratios[i],ratios[i],0,0)
        
        seq:add(tmp)
        seq:add(nn.GenNoise())

        seq:add(conv(net_input_depth, conv_num, 3,1))
        seq:add(bn(conv_num))
        seq:add(act(nil, inplace))

        seq:add(conv(conv_num, conv_num, 3,1))
        seq:add(bn(conv_num))
        seq:add(act(nil, inplace))

        seq:add(conv(conv_num, conv_num, 1,1))
        seq:add(bn(conv_num))
        seq:add(act(nil, inplace))



    if i == 1 then
        seq:add(nn.SpatialUpSamplingNearest(2))
        cur = seq
    else
        print(i)
        local cur_temp = cur

        cur = nn.Sequential()

        seq:add(bn(conv_num))
        cur_temp:add(bn(conv_num))

        cur:add(nn.Concat(2):add(cur_temp):add(seq))
        
        cur:add(conv(conv_num*i, conv_num*i, 3,1))
        cur:add(bn(conv_num*i))
        cur:add(act(nil, inplace))

        cur:add(conv(conv_num*i, conv_num*i, 3,1))
        cur:add(bn(conv_num*i))
        cur:add(act(nil, inplace))

        cur:add(conv(conv_num*i, conv_num*i, 1,1))
        cur:add(bn(conv_num*i))
        cur:add(act(nil, inplace))

        if i == #ratios then
            cur:add(conv(conv_num*i, 3, 1,1))
        else
            cur:add(nn.SpatialUpSamplingNearest(2)) 

        end
    end
end

model = cur
print (model)

return model

-- Decoder-encoder like model with skip-connections

local act = function() return nn.LeakyReLU(nil, true) end

local nums_3x3down = {16, 32, 64, 128, 128}
local nums_1x1 = {4, 4, 4, 4, 4}
local nums_3x3up = {16, 32, 64, 128, 128}

local model = nn.Sequential():add(nn.NoiseFill())
model_tmp = model


input_depth = net_input_depth
for i = 1,#nums_3x3down do
      
        local deeper = nn.Sequential()
        local skip = nn.Sequential()
        
        model_tmp:add(nn.Concat(2):add(skip):add(deeper))

        skip:add(conv(input_depth, nums_1x1[i], 1,1))
        skip:add(bn(nums_1x1[i]))
        skip:add(act())
        
        
        
        deeper:add(conv(input_depth, nums_3x3down[i], 3,2))
        deeper:add(bn(nums_3x3down[i]))
        deeper:add(act())

        deeper:add(conv(nums_3x3down[i], nums_3x3down[i], 3,1))
        deeper:add(bn(nums_3x3down[i]))
        deeper:add(act())

        deeper_main = nn.Sequential()

        if i == #nums_3x3down  then
            k = nums_3x3down[i]
        else
            deeper:add(deeper_main)
            k = nums_3x3up[i+1]
        end

        deeper:add(nn.SpatialUpSamplingNearest(2))

        deeper:add(bn(nums_3x3down[i]))
        skip:add(bn(nums_1x1[i]))
        

        model_tmp:add(conv(nums_1x1[i] +  k, nums_3x3up[i], 3,1))
        model_tmp:add(bn(nums_3x3up[i]))
        model_tmp:add(act())

        model_tmp:add(conv(nums_3x3up[i], nums_3x3up[i], 1))
        model_tmp:add(bn(nums_3x3up[i]))
        model_tmp:add(act())
        
        
        input_depth = nums_3x3down[i]
        model_tmp = deeper_main
end
model:add(conv(nums_3x3up[1], 3, 1,1))

return model

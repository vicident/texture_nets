-- Pyramid, but with tunable filter numbers.
-- This has more convenient filter numbers (more filters for low spatial resolution)

local ratios = {32, 16, 8, 4, 2, 1}
local filters_num = {128, 128, 64, 32, 16, 8}

local act = function() return nn.LeakyReLU(nil, true) end
local conv_num = 8



local cur = nil
for i = 1,#ratios do
        seq = nn.Sequential()

        local tmp =  nn.SpatialAveragePooling(ratios[i],ratios[i],ratios[i],ratios[i],0,0)
        
        seq:add(tmp)
        seq:add(nn.NoiseFill(num_noise_channels))

        seq:add(conv(net_input_depth, filters_num[i], 5))
        seq:add(bn(filters_num[i]))
        seq:add(act())

        seq:add(conv(filters_num[i], filters_num[i], 3))
        seq:add(bn(filters_num[i]))
        seq:add(act())

        seq:add(conv(filters_num[i], filters_num[i], 1))
        seq:add(bn(filters_num[i]))
        seq:add(act())



    if i == 1 then
        seq:add(nn.SpatialUpSamplingNearest(2))
        cur = seq
    else
        local cur_temp = cur

        cur = nn.Sequential()

        -- Batch norm before merging 
        seq:add(bn(filters_num[i - 1]))
        cur_temp:add(bn(filters_num[i - 1]))


        cur:add(nn.Concat(2):add(cur_temp):add(seq))
        
        cur:add(conv(filters_num[i-1] + filters_num[i], filters_num[i], 5))
        cur:add(bn(filters_num[i]))
        cur:add(act())

        cur:add(conv(filters_num[i], filters_num[i], 3))
        cur:add(bn(filters_num[i]))
        cur:add(act())

        cur:add(conv(filters_num[i], filters_num[i], 1))
        cur:add(bn(filters_num[i]))
        cur:add(act())

        

        if i == #ratios then
            cur:add(conv(filters_num[i], 3, 1))
        else
            cur:add(nn.SpatialUpSamplingNearest(2)) 
        end
    end
end
model = cur

return model

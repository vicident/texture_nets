require 'cutorch'
require 'nn'
require 'cunn'
require 'cudnn'
require 'image'
require 'src/utils.lua'

local cmd = torch.CmdLine()

cmd:option('-noise_depth', 3, 'Noise depth used to train model.')
cmd:option('-model', '', 'Path to trained model.')
cmd:option('-sample_size', 256)
cmd:option('-save_path', '.', 'Path to save samples.')

local params = cmd:parse(arg)

function deprocess_many(img)
    local temp = img:clone()
    for i=1,temp:size(1) do
        temp[i]:copy(torch.clamp(deprocess(temp[i]), 0, 1))
    end
    return temp
end

local model = torch.load(params.model):cuda()

local input = torch.zeros(4, params.noise_depth, params.sample_size, params.sample_size):cuda() 
local samples = model:forward(input):double()
local images = deprocess_many(samples)

for i =1,images:size(1) do
    image.save(params.save_path .. '/' .. i .. '.png', images[i])
end
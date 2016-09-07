require 'nn'
require 'image'
require 'InstanceNormalization'
require 'src/utils'

paths.dofile('util.lua')

local cmd = torch.CmdLine()

cmd:option('-input_image', '', 'Image to stylize.')
cmd:option('-image_size', 0, 'Resize input image to. Do not resize if 0.')
cmd:option('-model_t7', '', 'Path to trained model.')
cmd:option('-save_path', 'stylized.jpg', 'Path to save stylized image.')
cmd:option('-cpu', false, 'use this flag to run on CPU')
cmd:option('-binary', 0, 'Binary net flag')

params = cmd:parse(arg)

-- Load model from checkpoint and set type
local model = torch.load(params.model_t7)

-- Convert
saveParams(model, 'model.t7')

--[[
-- Load from converted
normalization = nn.SpatialBatchNormalization
pad = nn.SpatialReplicationPadding
model = torch.load(params.model_t7)
params.tv_weight = 0
params.mode = 'style'
backend = nn
local model = require('models/johnson')
loadParams(model, 'model.t7')
]]--
if params.cpu then 
  tp = 'torch.FloatTensor'
else
  require 'cutorch'
  require 'cunn'
  require 'cudnn'
  
  tp = 'torch.CudaTensor'
  model = cudnn.convert(model, cudnn)
end

model:type(tp)
model:evaluate()

local bNodes = {}

if params.binary then
  local nodes_all = model:listModules()
  -- Select all weighted layers
  lcnt = 0  
  for i=1,#nodes_all do
    if nodes_all[i].weight ~= nil then
      if nodes_all[i].weight:nDimension() >= 2 then
        table.insert(bNodes, nodes_all[i])
        lcnt = lcnt + 1
      end
    end
  end
end

print(string.format("%d layers have been binarized", #bNodes))

if #bNodes > 0 then
    -- Binarize weights in selected layers
    binarizeConvParms(bNodes)
end

saveParams(model, 'model.t7')

-- Load image and scale
local img = image.load(params.input_image, 3):float()
if params.image_size > 0 then
  img = image.scale(img, params.image_size, params.image_size)
end

-- Stylize
local input = img:add_dummy()
local stylized = model:forward(input:type(tp)):double()
stylized = deprocess(stylized[1])

--parameters:copy(realParams)

-- Save
image.save(params.save_path, torch.clamp(stylized,0,1))

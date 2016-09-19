require 'nn'
require 'image'
require 'InstanceNormalization'
require 'src/utils'
optnet = require 'optnet'
posix = require 'posix' 
timersub, gettimeofday = posix.timersub, posix.gettimeofday
paths.dofile('util.lua')

local cmd = torch.CmdLine()

cmd:option('-input_image', '', 'Image to stylize.')
cmd:option('-image_size', 0, 'Resize input image to. Do not resize if 0.')
cmd:option('-model_t7', '', 'Path to trained model.')
cmd:option('-save_path', 'stylized.jpg', 'Path to save stylized image.')
cmd:option('-cpu', false, 'use this flag to run on CPU')
cmd:option('-binary', false, 'Binary net flag')
cmd:option('-optnet', false, 'Optimize network memory')

params = cmd:parse(arg)

-- Load model from checkpoint and set type
--local model = torch.load(params.model_t7)
--print(model)
-- Convert
--saveParams(model, 'model.t7')

-- Load from converted
normalization = nn.InstanceNormalization
pad = nn.SpatialReplicationPadding
params.tv_weight = 0
params.mode = 'style'
backend = nn
local model = require('models/johnson')

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

--loadOptModel(model, 'model_o.t7')
loadParams(model, 'model.t7')

local bNodes = {}

if params.binary then
  local nodes_all = model:listModules()
  -- Select all weighted layers
  for i=1,#nodes_all do
    if nodes_all[i].weight ~= nil then
      if nodes_all[i].weight:nDimension() >= 2 then
        table.insert(bNodes, nodes_all[i])
      end
    end
  end
end

print(string.format("%d layers have been binarized", #bNodes))

if #bNodes > 0 then
    -- Binarize weights in selected layers
    binarizeConvParms(bNodes)
end

if params.optnet then
    input = torch.FloatTensor(1, 3, 256, 256)
    opts = {inplace=true, mode='inference', reuseBuffers=true, removeGradParams=true}
    optnet.optimizeMemory(model, input, opts)
    saveOptModel(model, 'model_o.t7')
end

-- Load image and scale
local img = image.load(params.input_image, 3):float()
if params.image_size > 0 then
  img = image.scale(img, params.image_size, params.image_size)
end

-- Stylize
local input = img:add_dummy()
local start_time = gettimeofday()
local stylized = model:forward(input:type(tp)):double()
local elapsed = timersub(gettimeofday(), start_time)
print(string.format('Forward took: %.0fms\n', elapsed.sec * 1000 + elapsed.usec / 1000))
stylized = deprocess(stylized[1])

--parameters:copy(realParams)

-- Save
image.save(params.save_path, torch.clamp(stylized,0,1))

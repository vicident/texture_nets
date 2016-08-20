require 'torch'
require 'nn'
require 'image'
require 'optim'

require 'cutorch'
require 'cunn'

require 'src/utils'
require 'src/descriptor_net'

local DataLoader = require 'dataloader'

use_display, display = pcall(require, 'display')
if not use_display then 
  print('torch.display not found. unable to plot') 
end

----------------------------------------------------------
-- Parameters
----------------------------------------------------------
local cmd = torch.CmdLine()

cmd:option('-content_layers', 'relu4_2', 'Layer to attach content loss. Only one supported for now.')
cmd:option('-style_layers', 'relu1_1,relu2_1,relu3_1,relu4_1', 'Layer to attach content loss. Only one supported for now.')

cmd:option('-learning_rate', 1e-3)

cmd:option('-num_iterations', 50000)
cmd:option('-save_every', 1000)
cmd:option('-batch_size', 1)

cmd:option('-image_size', 256)

cmd:option('-content_weight',1)
cmd:option('-style_weight', 1)
cmd:option('-tv_weight', 0, 'Total variation weight.')

cmd:option('-style_image', '')
cmd:option('-style_size', 256)

cmd:option('-mode', 'style', 'style|texture')

cmd:option('-checkpoints_path', 'data/checkpoints/', 'Directory to store intermediate results.')
cmd:option('-model', 'pyramid', 'Path to generator model description file.')

cmd:option('-normalize_gradients', 'false', 'L1 gradient normalization inside descriptor net. ')
cmd:option('-vgg_no_pad', 'false')
cmd:option('-normalization', 'instance', 'batch|instance')

cmd:option('-proto_file', 'data/pretrained/VGG_ILSVRC_19_layers_deploy.prototxt', 'Pretrained')
cmd:option('-model_file', 'data/pretrained/VGG_ILSVRC_19_layers.caffemodel')
cmd:option('-backend', 'cudnn', 'nn|cudnn')

-- Dataloader
cmd:option('-dataset', 'style')
cmd:option('-data', '', 'Path to dataset. Structure like in fb.resnet.torch repo.')
cmd:option('-manualSeed', 0)
cmd:option('-nThreads', 4, 'Data loading threads.')

params = cmd:parse(arg)

params.normalize_gradients = params.normalize_gradients ~= 'false'
params.vgg_no_pad = params.vgg_no_pad ~= 'false'
params.circular_padding = params.circular_padding ~= 'false'

-- For compatibility with Justin Johnsons code
params.texture_weight = params.style_weight
params.texture_layers = params.style_layers
params.texture = params.style_image

if params.normalization == 'instance' then
  require 'InstanceNormalization'
  normalization = nn.InstanceNormalization
elseif params.normalization == 'batch' then
  normalization = nn.SpatialBatchNormalization
end

if params.mode == 'texture' then
	params.content_layers = ''
  pad = nn.SpatialCircularPadding
	-- Use circular padding
	conv = convc
else
  pad = nn.SpatialReplicationPadding
end

trainLoader, valLoader = DataLoader.create(params)

if params.backend == 'cudnn' then
  require 'cudnn'
  cudnn.fastest = true
  cudnn.benchmark = true
  backend = cudnn
else
  backend = nn
end

-- Define model
local net = require('models/' .. params.model):cuda()

local crit = nn.ArtisticCriterion(params)

----------------------------------------------------------
-- feval
----------------------------------------------------------


local iteration = 0

local parameters, gradParameters = net:getParameters()
local loss_history = {}
function feval(x)
  iteration = iteration + 1

  if x ~= parameters then
      parameters:copy(x)
  end
  gradParameters:zero()
  
  local loss = 0
  
  -- Get batch 
  local images = trainLoader:get()

  target_for_display = images.target
  local images_target = preprocess1(images.target):cuda()
  local images_input = images.input:cuda()-0.5

  -- Forward
  local out = net:forward(images_input)
  loss = loss + crit:forward({out, images_target})
  
  -- Backward
  local grad = crit:backward({out, images_target}, nil)
  net:backward(images_input, grad[1])

  loss = loss/params.batch_size
  
  table.insert(loss_history, {iteration,loss})
  print('#it: ', iteration, 'loss: ', loss)
  return loss, gradParameters
end

----------------------------------------------------------
-- Optimize
----------------------------------------------------------
print('        Optimize        ')

style_weight_cur = params.style_weight
content_weight_cur = params.content_weight

local optim_method = optim.adam
local state = {
   learningRate = params.learning_rate,
}

for it = 1, params.num_iterations do

  -- Optimization step
  optim_method(feval, parameters, state)

  -- Visualize
  if it%50 == 0 then
    collectgarbage()

    local output = net.output:double()
    local imgs  = {}
    for i = 1, output:size(1) do
      local img = deprocess(output[i])
      table.insert(imgs, torch.clamp(img,0,1))
    end
    if use_display then 
      display.image(target_for_display, {win=1, width=512,title = 'Target'})
      display.image(imgs, {win=0, width=512})
      display.plot(loss_history, {win=2, labels={'iteration', 'Loss'}})
    end
  end
  
  if it%2000 == 0 then 
    state.learningRate = state.learningRate*0.8
  end

  -- Dump net
  if it%params.save_every == 0 or it == params.num_iterations then 
    local net_to_save = cudnn.convert(deepCopy(net):float():clearState(), nn)
    torch.save(paths.concat(params.checkpoints_path, 'model_' .. it .. '.t7'), net_to_save)
  end
end

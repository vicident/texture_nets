require 'torch'
require 'cutorch'
require 'nn'
require 'cunn'
require 'image'
require 'optim'
require 'hdf5'

display = require('display')

require 'src/utils.lua'
require 'src/descriptor_net.lua'

local cmd = torch.CmdLine()

cmd:option('-style_layers', 'relu1_1,relu2_1,relu3_1,relu4_1,relu5_1', 'Layers to attach texture (style) loss.')
cmd:option('-content_layers', 'relu4_2', 'Layer to attach content loss. Only one supported for now.')

cmd:option('-content_weight', 6e-1)
cmd:option('-style_weight', 1e3)

cmd:option('-style_image', 'data/textures/red-peppers256.o.jpg')
cmd:option('-train_hdf5', 'data/256.hdf5')
cmd:option('-train_images_path', 'path/to/imagenet_val', 'Just because I did not find a way to iterate through data in hdf5 without names.')

cmd:option('-learning_rate', 1e-1)
cmd:option('-num_iterations', 3000)

cmd:option('-batch_size', 16)
cmd:option('-image_size', 256)
cmd:option('-noise_depth', 3, 'How many noise channels to append to image.')

cmd:option('-gpu', 0, 'Zero indexed gpu number.')
cmd:option('-tmp_path', 'data/out/', 'Directory to store intermediate results.')
cmd:option('-model_name', '', 'Path to generator model description file.')

cmd:option('-normalize_gradients', false, 'L1 gradient normalization inside descriptor net. ')
cmd:option('-vgg_no_pad', false)

cmd:option('-proto_file', 'data/pretrained/VGG_ILSVRC_19_layers_deploy.prototxt', 'Pretrained')
cmd:option('-model_file', 'data/pretrained/VGG_ILSVRC_19_layers.caffemodel')
cmd:option('-backend', 'nn', 'nn|cudnn')

cmd:option('-no_circ', false, 'Whether to use circular padding for convolutions.')

params = cmd:parse(arg)

-- For compatibility with Justin Johnsons code
params.texture_weight = params.style_weight
params.texture_layers = params.style_layers
params.texture = params.style_image

if params.backend == 'cudnn' then
  require 'cudnn'
  -- cudnn.benchmark = true
  backend = cudnn
else
  backend = nn
end

-- Whether to use circular padding
if not params.no_circ then
  conv = convc
end

cutorch.setDevice(params.gpu+1)

-- Input dim
net_input_depth = 3 + params.noise_depth
num_noise_channels = params.noise_depth

-- Define model
net = require('models/' .. params.model_name):cuda()

-- Setup descriptor net
local descriptor_net, content_losses, texture_losses = create_loss_net(params)

----------------------------------------------------------
-- Batch generator
----------------------------------------------------------

-- Collect image names 
image_names = {}
for f in paths.files(params.train_images_path, 'JPEG') do
  table.insert(image_names, f)
end

local train_hdf5 = hdf5.open(params.train_hdf5, 'r')

-- allocate reusable space
inputs_batch = torch.Tensor(params.batch_size, net_input_depth, params.image_size, params.image_size)
contents_batch = torch.Tensor(params.batch_size, 512, params.image_size/8, params.image_size/8)

cur_index_train = 1 
function get_input_train()
  -- ignore last for simplicity
  if cur_index_train > #image_names - params.batch_size then
    cur_index_train = 1 
  end

  for i = 0, params.batch_size-1 do
    contents_batch[i+1] = train_hdf5:read(image_names[cur_index_train + i]..  '_content'):all()
    inputs_batch:narrow(2,1,3)[i+1] = train_hdf5:read(image_names[cur_index_train + i] ..  '_image' ):all()
  end
  cur_index_train = cur_index_train + params.batch_size

  return inputs_batch:cuda(), contents_batch:cuda() 
end

----------------------------------------------------------
-- feval
----------------------------------------------------------

iteration = 0

-- dummy storage, this will not be changed during training
inputs_batch = torch.Tensor(params.batch_size, net_input_depth, params.image_size, params.image_size):uniform():cuda()

local parameters, gradParameters = net:getParameters()
loss_history = {}
function feval(x)
  iteration = iteration + 1

  if x ~= parameters then
      parameters:copy(x)
  end
  gradParameters:zero()
  
  -- get batch 
  local images, contents = get_input_train()  
  
  -- set current `relu4_2` content 
  content_losses[1].target = contents

  -- forward
  local out = net:forward(images)
  descriptor_net:forward(out)
  
  -- backward
  local grad = descriptor_net:backward(out, nil)
  net:backward(images, grad)
  
  -- collect loss
  local loss = 0
  for _, mod in ipairs(texture_losses) do
    loss = loss + mod.loss
  end
  for _, mod in ipairs(content_losses) do
    loss = loss + mod.loss
  end
  table.insert(loss_history, {iteration,loss/params.batch_size})
  print(iteration, loss/params.batch_size)

  return loss, gradParameters
end

----------------------------------------------------------
-- Optimize
----------------------------------------------------------
print('        Optimize        ')

optim_method = optim.adam
state = {
   learningRate = params.learning_rate,
}

for it = 1, params.num_iterations do
  
  -- Optimization step
  optim_method(feval, parameters, state)

  -- Visualize
  if it%10 == 0 then
    collectgarbage()

    local output = net.output:clone():double()

    local imgs  = {}
    for i = 1, output:size(1) do
      local img = deprocess(output[i])
      table.insert(imgs, torch.clamp(img,0,1))
      image.save(params.tmp_path .. 'train' .. i .. '_' .. it .. '.png',img)
    end

    display.image(imgs, {win=params.gpu, width=params.image_size*3,title = params.gpu})
    display.plot(loss_history, {win=params.gpu+4, labels={'iteration', 'Loss'}, title='Gpu ' .. params.gpu .. ' Loss'})
  end
  
  if it%300 == 0 then 
    state.learningRate = state.learningRate*0.8
  end

  -- Dump net, the file is huge
  if it%200 == 0 then 
    torch.save(params.tmp_path .. 'model.t7', net)
  end
end
-- Clean net and dump it, ~ 500 kB
netLighter(net)
torch.save(params.tmp_path .. 'model.t7', net)

train_hdf5:close()
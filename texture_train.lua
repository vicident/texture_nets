require 'torch'
require 'cutorch'
require 'nn'
require 'cunn'
require 'cudnn'
require 'image'
require 'optim'

display = require('display')
dofile 'src/utils.lua'

local cmd = torch.CmdLine()

cmd:option('-texture_layers', 'relu1_1,relu2_1,relu3_1,relu4_1,relu5_1', 'Layers to attach texture loss.')
-- cmd:option('-style_layers', 'relu1_1,relu1_2,relu2_1,relu2_2,relu3_1,relu3_2,relu3_3,relu3_4,relu4_1,relu4_2,relu4_3,relu4_4,relu5_1')

cmd:option('-texture', 'data/style/pleades.jpg','Style target image')

cmd:option('-learning_rate', 1e-1)
cmd:option('-num_iterations', 1000)

cmd:option('-batch_size', 16)

cmd:option('-image_size', 256)
cmd:option('-gpu', 2, 'Zero indexed gpu number.')
cmd:option('-tmp_path', 'data/out/', 'Directory to store intermediate results.')
cmd:option('-model_name', '', 'Path to generator model description file.')

cmd:option('-normalize_gradients', false)
cmd:option('-vgg_no_pad', false)

cmd:option('-proto_file', 'data/pretrained/VGG_ILSVRC_19_layers_deploy.prototxt')
cmd:option('-model_file', 'data/pretrained/VGG_ILSVRC_19_layers.caffemodel')
cmd:option('-backend', 'nn', 'nn|cudnn')

cmd:option('-circ', false, 'Whether to use circular padding for convolutions.')

params = cmd:parse(arg)
params.texture_weight = 1

if params.backend == 'cudnn' then
  cudnn.benchmark = true
end

if params.circ then
  conv = convc
end

if params.style_size == 0 then
  params.style_size =  params.image_size
end
cutorch.setDevice(params.gpu+1)

net_input_depth = 3

------------------------------------------------------------
-- Load dataset
------------------------------------------------------------
-- local style_image = image.load(params.style_image, 3)
-- style_image = image.scale(style_image, params.image_size, params.image_size, 'bilinear')

----------------------------------------------------------
-- Define model
----------------------------------------------------------
net = require('models/' .. params.model_name):cuda()

-- local sample_image = image.load(train_images_paths[1], 3)
-- sample_image = image.scale(sample_image, params.image_size, params.image_size, 'bilinear'):add_dummy()

-- local sample = net:forward(sample_image:cuda())
-- params.vgg_in_size = sample[1]:size(3)
params.vgg_in_size = 256

print ("------image size--------",  params.vgg_in_size)
----------------------------------------------------------
-- Setup VGG
----------------------------------------------------------

dofile 'src/descriptor_net.lua'

local descriptor_net, _, texture_losses = create_loss_net(params)

counter = 0
cur_index_train = 1 
cur_index_test = 1 
inputs_batch = torch.Tensor(params.batch_size, net_input_depth, params.image_size, params.image_size):uniform():cuda()
contents_batch = torch.Tensor(params.batch_size, 512, params.image_size/8, params.image_size/8):uniform():cuda()
-- contents_batch = torch.Tensor(params.batch_size, 512, 23, 23)

----------------------------------------------------

local parameters, gradParameters = net:getParameters()
lossss = {}
function feval(x)

    local start_time = socket.gettime()
    -- timer = torch.Timer()
    local images, contents = inputs_batch, contents_batch
    local end_time = socket.gettime()
    local t1 = end_time - start_time
    -- local t1 = timer:time().real
    timer = torch.Timer()
   

    counter = counter + 1
    if x ~= parameters then
        parameters:copy(x)
    end
    gradParameters:zero()
    
    -- forward
    local out = net:forward(images)
    descriptor_net:forward(out)
    
    -- backward
    local grad = descriptor_net:backward(out, nil)
    net:backward(images, grad)
    
    local loss = 0

    for _, mod in ipairs(texture_losses) do
      loss = loss + mod.loss
    end
    
    table.insert(lossss, {counter,loss/params.batch_size })
    print(counter, loss/params.batch_size)
    return loss, gradParameters
end

----------------------------
--             Optimize
----------------------------
print('        Optimize        ')


optim_method = optim.adam
state = {
   learningRate = params.learning_rate,
}


for t = 1, params.num_iterations do
    optim_method(feval, parameters, state)

    if t%10 == 0 then
      collectgarbage()

      -- local images, contents = get_input_train()

      local output_train = net.modules[#net.modules].output:double()


      local imgs  = {}
      for i = 1, output_train:size(1) do
        local img = deprocess(output_train[i])
        table.insert(imgs, img)
        image.save(params.tmp_path .. 'train' .. i .. '_' .. t .. '.png',img)
      end

      display.image(imgs, {win=params.gpu, width=params.vgg_in_size*3,title = params.gpu})
      display.plot(lossss, {win=params.gpu+4, labels={'epoch', 'Loss'}, title='Gpu ' .. params.gpu .. ' Loss'})
    end
    
    if t>=1000 and t%200 == 0 then 
      state.learningRate = state.learningRate*0.7 
      -- torch.save(params.tmp_path .. 'model.t7', net)
    end

    if t%200 == 0 then 
      torch.save(params.tmp_path .. 'model.t7', net)
    end
end
torch.save(params.tmp_path .. 'model.t7', net)
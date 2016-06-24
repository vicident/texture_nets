require 'cutorch'
require 'nn'
require 'cunn'
require 'image'
require 'hdf5'
require 'xlua'
require 'src/utils'
require 'loadcaffe'

---------------------------------------------------------
-- Define params
---------------------------------------------------------
local cmd = torch.CmdLine()

cmd:option('-images_path', 'path/to/imagenet_val')
cmd:option('-save_to', 'data/256.hdf5')
cmd:option('-resize_to', 256)

cmd:option('-vgg_no_pad', false , 'Whether to use padding in convolutions in descriptor net.')
cmd:option('-gpu', 0)
cmd:option('-backend', 'nn', 'nn|cudnn')
cmd:option('-proto_file', 'data/pretrained/VGG_ILSVRC_19_layers_deploy.prototxt')
cmd:option('-model_file', 'data/pretrained/VGG_ILSVRC_19_layers.caffemodel')

local cmd_params = cmd:parse(arg)


---------------------------------------------------------
-- Parse params
---------------------------------------------------------

if cmd_params.backend == 'cudnn' then
  require 'cudnn'
  cudnn.benchmark = true
  print('cudnn')
end
cutorch.setDevice(cmd_params.gpu+1)

---------------------------------------------------------
-- Define helpers
---------------------------------------------------------

function load_image(image_path, scale_to)
  local img = image.load(image_path, 3)
  img = image.scale(img, scale_to, scale_to, 'bilinear')

  return img:cuda(), preprocess(img):cuda()
end


function load_vgg(cmd_params)
  local vgg = loadcaffe.load(cmd_params.proto_file, cmd_params.model_file, cmd_params.backend):cuda()

  print ('Leaving only 23 modules (till relu4_2')
  for i=1,23 do
    vgg:remove()
  end

  if cmd_params.vgg_no_pad then

    if cmd_params.backend == 'nn' then
        conv_modules = vgg:findModules('nn.SpatialConvolution')
    else
        conv_modules = vgg:findModules('cudnn.SpatialConvolution')
    end

    for _, module in ipairs(conv_modules) do
      module.padW = 0 
      module.padH = 0 
    end

  end
  return vgg
end

---------------------------------------------------------
-- Extract content 
---------------------------------------------------------
vgg = load_vgg(cmd_params)

-- File to save to
local out_file = hdf5.open(cmd_params.save_to, 'w')

-- Get list of images
local path_generator = paths.files(cmd_params.images_path, is_image)
local images = {}
for image_name in path_generator do
  table.insert(images, image_name)
end

-- Go
for i, image_name in ipairs(images) do
  
  local img, img_preprocessed = load_image(cmd_params.images_path ..'/' .. image_name, cmd_params.resize_to)
  local content = vgg:forward(img_preprocessed):clone()      
  
  -- Store content and image
  out_file:write(image_name .. '_content', content:float())
  out_file:write(image_name .. '_image', img:float())
  
  if (i%500) == 0 then
    collectgarbage()
  end

  xlua.progress(i,#images)
end
out_file:close()

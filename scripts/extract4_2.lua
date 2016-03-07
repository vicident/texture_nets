require 'torch'
require 'cutorch'
require 'nn'
require 'cunn'
require 'cudnn'
require 'image'
require 'optim'
require 'hdf5'
cudnn.benchmark = true

function preprocess(img)
  local mean_pixel = torch.DoubleTensor({103.939, 116.779, 123.68})
  local perm = torch.LongTensor{3, 2, 1}
  img = img:index(1, perm):mul(256.0)
  mean_pixel = mean_pixel:view(3, 1, 1):expandAs(img)
  img:add(-1, mean_pixel)
  return img
end

function load_image(image_path, scale_to)
  local img = image.load(image_path, 3)
  img = image.scale(img, scale_to, scale_to, 'bilinear')

  return img:cuda(), preprocess(img):cuda()
end


local cmd = torch.CmdLine()


cmd:option('-images_path', '/home/cygnus/ILSVRC2012_img_val')
cmd:option('-save_to', 'data/512.hdf5')
cmd:option('-resize_to', 512)

cmd:option('-vgg_no_pad', false)
cmd:option('-gpu', 0)
cmd:option('-backend', 'cudnn')

local cmd_params = cmd:parse(arg)

image_size = cmd_params.image_size

cutorch.setDevice(cmd_params.gpu+1)

proto_file='data/models/VGG_ILSVRC_19_layers_deploy.prototxt'
model_file='data/models/VGG_ILSVRC_19_layers.caffemodel'


-- fails without this line..
image.load('/home/cygnus/ILSVRC2012_img_train/n02106030/n02106030_10330.JPEG', 3)

loadcaffe_wrap = require '../src/loadcaffe_wrapper'

function load_vgg()
  local vgg = loadcaffe_wrap.load(proto_file, model_file, cmd_params.backend):cuda()

  print ('Leaving only 23 modules')
  for i=1,23 do
    vgg:remove()
  end

  

  if cmd_params.vgg_no_pad then

    conv_modules = vgg:findModules('cudnn.SpatialConvolution')
    for _, module in ipairs(conv_modules) do
      module.padW = 0 
      module.padH = 0 
    end
  end

  -- for i=1, #vgg do
  --   layer_type = torch.type(vgg.modules[i])
  --   if (layer_type == 'nn.SpatialConvolution' or layer_type == 'nn.SpatialConvolutionMM' or layer_type == 'cudnn.SpatialConvolution') then
  --       vgg.modules[i].padW = 0 
  --       vgg.modules[i].padH = 0 
  --   end
  -- end

  return vgg
end

vgg = load_vgg()





-- files = paths.files('/home/cygnus/ILSVRC2012_img_val','.JPEG')
-- for image_name in files do
--   print (image_name)

--   local image_ = load_image('/home/cygnus/ILSVRC2012_img_val/' .. image_name, image_size):cuda()

--   local myFile = hdf5.open(save_dir .. image_name .. '.hdf5', 'w')
--   local options = hdf5.DataSetOptions()
--   -- options:setChunked(32, 32, 32)
--   -- options:setDeflate()

--   content = vgg:forward(image_):clone()      
--   myFile:write('content', content:float())
  
--   myFile:close()
-- end




-- local myFile = hdf5.open(save_dir .. 'data' .. '.hdf5', 'w')

-- files = paths.files('/home/cygnus/ILSVRC2012_img_val','.JPEG')
-- for image_name in files do
--   print (image_name)

--   local image_ = load_image('/home/cygnus/ILSVRC2012_img_val/' .. image_name, image_size):cuda()

--     local options = hdf5.DataSetOptions()
--   -- options:setChunked(32, 32, 32)
--   -- options:setDeflate()

--   content = vgg:forward(image_):clone()      
--   myFile:write(image_name, content:float())
  
  
-- end
-- myFile:close()


local out_file = hdf5.open(cmd_params.save_to, 'w')

files = paths.files(cmd_params.images_path, '.JPEG')
local  i = 1
for image_name in files do
  print (i)

  local img, img_preprocessed = load_image(cmd_params.images_path ..'/' .. image_name, cmd_params.resize_to)

  local content = vgg:forward(img_preprocessed):clone()      
  out_file:write(image_name .. '_content', content:float())
  out_file:write(image_name .. '_image', img:float())
  
  i = i + 1
  if (i%500) == 0 then
    collectgarbage()
  end
  
  if i > 20000 then
    break
  end
end
out_file:close()
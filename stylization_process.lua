require 'cutorch'
require 'nn'
require 'cunn'
require 'cudnn'
require 'image'
require 'src/utils'

local cmd = torch.CmdLine()

cmd:option('-input_image', '', 'Image to stylize.')
cmd:option('-image_size', 0, 'Resize input image to. Do not resize if 0.')

cmd:option('-model', '', 'Path to trained model.')
cmd:option('-noise_depth', 3, 'Noise depth used to train model.')
cmd:option('-gpu', 0, 'Zero indexed gpu.')

cmd:option('-save_path', 'stylized.png', 'Path to save stylized image.')

local params = cmd:parse(arg)

cutorch.setDevice(params.gpu+1)

-- Load model
local model = torch.load(params.model):cuda()
--model:evaluate()

-- Load image and scale
local img = image.load(params.input_image, 3)
if params.image_size > 0 then
  img = image.scale(img, params.image_size, params.image_size)
end

-- Create input tensor
local input = torch.FloatTensor(1, 3 + params.noise_depth, img:size(2), img:size(3))
input[1]:narrow(1, 1, 3):copy(img)

-- Stylize
local stylized = model:forward(input:cuda()):double()
-- stylized = torch.clamp(deprocess(stylized[1]), 0, 1)
stylized = deprocess(stylized[1])

-- Save
image.save(params.save_path, stylized)

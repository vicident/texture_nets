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
cmd:option('-save_path', 'stylized.jpg', 'Path to save stylized image.')

local params = cmd:parse(arg)

-- Load model
local model = torch.load(params.model):cuda()
print(model)
model = cudnn.convert(model, cudnn)

-- Load image and scale
local img = image.load(params.input_image, 3)
if params.image_size > 0 then
  img = image.scale(img, params.image_size, params.image_size)
end

-- Create input tensor
local input = torch.FloatTensor(1, 3, img:size(2), img:size(3))
input[1]:copy(img)

-- Stylize
local stylized = model:forward(input:cuda()):double()
print(stylized:size())
stylized = deprocess(stylized[1])

-- Save
image.save(params.save_path, stylized)

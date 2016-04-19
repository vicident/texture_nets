require 'src/content_loss'
require 'src/texture_loss'

require 'loadcaffe'

function nop()
  -- nop.  not needed by our net
end

function create_descriptor_net()
    
  local cnn = loadcaffe.load(params.proto_file, params.model_file, params.backend):cuda()

  -- load texture
  local texture_image = image.load(params.texture, 3)

  texture_image = image.scale(texture_image, params.image_size, 'bilinear')
  local texture_image = preprocess(texture_image):cuda():add_dummy()

  params.content_layers = params.content_layers or ''

  local content_layers = params.content_layers:split(",") 
  local texture_layers   = params.texture_layers:split(",")

  -- Set up the network, inserting texture and content loss modules
  local content_losses, texture_losses = {}, {}
  local next_content_idx, next_texture_idx = 1, 1
  local net = nn.Sequential()

  for i = 1, #cnn do
    if next_content_idx <= #content_layers or next_texture_idx <= #texture_layers then
      local layer = cnn:get(i)
      local name = layer.name
      local layer_type = torch.type(layer)
      local is_pooling = (layer_type == 'cudnn.SpatialMaxPooling' or layer_type == 'nn.SpatialMaxPooling')
      
      if layer_type == 'nn.SpatialConvolution' or layer_type == 'nn.SpatialConvolutionMM' or layer_type == 'cudnn.SpatialConvolution' then
        layer.accGradParameters = nop
      end
      
      if params.vgg_no_pad and (layer_type == 'nn.SpatialConvolution' or layer_type == 'nn.SpatialConvolutionMM' or layer_type == 'cudnn.SpatialConvolution') then
          print (name, ': padding set to 0')

          layer.padW = 0 
          layer.padH = 0 
      end

      net:add(layer)
   
      ---------------------------------
      -- Content
      ---------------------------------
      if name == content_layers[next_content_idx] then
        print("Setting up content layer", i, ":", layer.name)

        local this_contents = {}

        local target = torch.Tensor()

        local norm = false
        local loss_module = nn.ContentLoss(params.content_weight, target, norm):cuda()
        net:add(loss_module)
        table.insert(content_losses, loss_module)
        next_content_idx = next_content_idx + 1
      end

      ---------------------------------
      -- Texture
      ---------------------------------
      if name == texture_layers[next_texture_idx] then
        print("Setting up texture layer  ", i, ":", layer.name)
        local gram = GramMatrix():cuda()

        local target_features = net:forward(texture_image):clone()
        local target = gram:forward(nn.View(-1):cuda():setNumInputDims(2):forward(target_features[1])):clone()

        target:div(target_features[1]:nElement())

        local norm = params.normalize_gradients
        local loss_module = nn.TextureLoss(params.texture_weight, target, norm):cuda()
        
        net:add(loss_module)
        table.insert(texture_losses, loss_module)
        next_texture_idx = next_texture_idx + 1
      end
    end
  end

  net:add(nn.DummyGradOutput())

  -- We don't need the base CNN anymore, so clean it up to save memory.
  cnn = nil
  for i=1,#net.modules do
    local module = net.modules[i]
    if torch.type(module) == 'nn.SpatialConvolutionMM' or torch.type(module) == 'nn.SpatialConvolution' or torch.type(module) == 'cudnn.SpatialConvolution' then
        module.gradWeight = nil
        module.gradBias = nil
    end
  end
  collectgarbage()
      
  return net, content_losses, texture_losses
end

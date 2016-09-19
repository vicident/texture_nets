
function zeroBias(nodes)
   for i =1, #nodes do
    local n = nodes[i].bias:fill(0)
   end
end


function updateBinaryGradWeight(nodes)
   for i=1,#nodes do
    local n = nodes[i].weight[1]:nElement()
    local s = nodes[i].weight:size()
    local m = nodes[i].weight:norm(1,4):sum(3):sum(2):div(n):expand(s);
    m[nodes[i].weight:le(-1)]=0;
    m[nodes[i].weight:ge(1)]=0;
    m:add(1/(n)):mul(1-1/s[2]):mul(n);
    nodes[i].gradWeight:cmul(m)--:cmul(mg)
   end
end


function meancenterConvParms(nodes)
   for i=1,#nodes do
    local s = nodes[i].weight:size()
    local negMean = nodes[i].weight:mean(2):mul(-1):repeatTensor(1,s[2],1,1);  
    nodes[i].weight:add(negMean)
   end
end


function binarizeConvParms(nodes)
   for i=1,#nodes do
    local n = nodes[i].weight[1]:nElement()
    local s = nodes[i].weight:size()
    local m = nodes[i].weight:norm(1,4):sum(3):sum(2):div(n);
    nodes[i].weight:sign():cmul(m:expand(s))
   end
end


function clampConvParms(nodes)
   for i=1,#nodes do
    nodes[i].weight:clamp(-1,1)
   end
end


-- Save weights, biases and statistics
function saveParams(model, path)

    local weights = {}
    local biases = {}
    local bn_dict = {}
    -- Select all modules in network
    local nodes = model:listModules()
    print ('num modules:', #nodes)
    local m = 1
    local n = 1
    for i=1,#nodes do
        -- Choose only layers with weights
        if nodes[i].weight ~= nil then
            weights[m] = torch.FloatTensor(nodes[i].weight:size()):copy(nodes[i].weight)
            biases[m] = torch.FloatTensor(nodes[i].bias:size()):copy(nodes[i].bias)
            m = m + 1
        end
        if nodes[i].running_mean ~= nil then
            bn_dict[n] = {}
            bn_dict[n]['mean'] = torch.FloatTensor(nodes[i].running_mean:size()):copy(nodes[i].running_mean)
            bn_dict[n]['var'] = torch.FloatTensor(nodes[i].running_var:size()):copy(nodes[i].running_var)
            n = n + 1
        end
    end 

    print ('num batch norm:', n-1)
    print ('num weighted:', m-1)

    local model_2_save = {} 
    model_2_save['weights'] = weights
    model_2_save['biases'] = biases
    model_2_save['bn'] = bn_dict

    -- Serialize model to file
    torch.save(path, model_2_save)
end


-- Load parameters to nn model with binary layers
function loadParams(model, path)
    --Select all modules in network
    local nodes = model:listModules()
    print('num modules:', #nodes)
    local m = 1
    local n = 1
    -- Load data map 
    local saved_model = torch.load(path)

    local saved_weights = saved_model['weights']
    local saved_biases = saved_model['biases']
    local saved_bn = saved_model['bn']

    -- Load weights and biases
    for i=1,#nodes do
        if nodes[i].weight ~= nil then
            nodes[i].weight:copy(saved_weights[m])
            nodes[i].bias:copy(saved_biases[m])
            m = m + 1
        end
        if nodes[i].running_mean ~= nil then
            nodes[i].running_mean:copy(saved_bn[n]['mean'])
            nodes[i].running_var:copy(saved_bn[n]['var'])
            n = n + 1
        end
    end

    print ('num batch norm:', n-1)
    print ('num weighted:', m-1)

end

function saveOptModel(model, path)
    local weights = {}
    local biases = {}
    local bn_dict = {}
    local storage_ptrs = {}
    local storage_sizes = {}
    local tensor_map = {}
    local tensor_sizes = {}
    -- Select all modules in network
    local nodes = model:listModules()
    print ('num modules:', #nodes)
    local m = 1
    local n = 1
    local p = 1
    local s = 1

    local function collect_ptrs(self) 
      tensor_map[s] = {}
      tensor_sizes[s] = {}
      for k, v in pairs(self) do
        if torch.isTensor(v) and v:storage() then
          local ptr = torch.pointer(v:storage())
          if storage_ptrs[ptr] == nil then
            storage_ptrs[ptr] = p
            storage_sizes[p] = v:storage():size()
            --print('storage type:', torch.type(v:storage()))
            p = p + 1
          end
          tensor_map[s][k] = storage_ptrs[ptr]
          tensor_sizes[s][k] = v:size()
          --print('tensor type:', torch.type(v))
        end
      end
      s = s + 1
    end

    model:apply(collect_ptrs)
    --print("ptrs:", tensor_ptrs)
    print("#pointers:", p-1)
    print("#modules:", s-1)

    for i=1,#nodes do
        
        -- Choose only layers with weights
        if nodes[i].weight ~= nil then
            weights[m] = torch.FloatTensor(nodes[i].weight:size()):copy(nodes[i].weight)
            biases[m] = torch.FloatTensor(nodes[i].bias:size()):copy(nodes[i].bias)
            m = m + 1
        end
        if nodes[i].running_mean ~= nil then
            bn_dict[n] = {}
            bn_dict[n]['mean'] = torch.FloatTensor(nodes[i].running_mean:size()):copy(nodes[i].running_mean)
            bn_dict[n]['var'] = torch.FloatTensor(nodes[i].running_var:size()):copy(nodes[i].running_var)
            n = n + 1
        end
    end 

    print ('num batch norm:', n-1)
    print ('num weighted:', m-1)

    local model_2_save = {} 
    model_2_save['weights'] = weights
    model_2_save['biases'] = biases
    model_2_save['bn'] = bn_dict
    model_2_save['th_map'] = tensor_map
    model_2_save['th_sizes'] = tensor_sizes
    model_2_save['st_sizes'] = storage_sizes

    -- Serialize model to file
    torch.save(path, model_2_save)
end


function loadOptModel(model, path)
    --Select all modules in network
    local nodes = model:listModules()
    print('num modules:', #nodes)
    local m = 1
    local n = 1
    -- Load data map 
    local saved_model = torch.load(path)

    local saved_weights = saved_model['weights']
    local saved_biases = saved_model['biases']
    local saved_bn = saved_model['bn']

    local storage_repeat = {}
    local storage_sizes = saved_model['st_sizes']
    local tensor_sizes = saved_model['th_sizes']
    local tensor_map = saved_model['th_map']
    local s = 1

    local function collect_ptrs(self) 
      for k, v in pairs(tensor_map[s]) do
        if storage_repeat[v] == nil then
          storage_repeat[v] = torch.FloatStorage(storage_sizes[v])
        end
        self[k] = torch.FloatTensor(storage_repeat[v], 1, tensor_sizes[s][k])
      end
      s = s + 1
    end

    model:apply(collect_ptrs)
 
    -- Load weights and biases
    for i=1,#nodes do
        if nodes[i].weight ~= nil then
            nodes[i].weight:copy(saved_weights[m])
            nodes[i].bias:copy(saved_biases[m])
            m = m + 1
        end
        if nodes[i].running_mean ~= nil then
            nodes[i].running_mean:copy(saved_bn[n]['mean'])
            nodes[i].running_var:copy(saved_bn[n]['var'])
            n = n + 1
        end
    end
    
    collectgarbage()

    print('#modules:', s-1)    
    print ('num batch norm:', n-1)
    print ('num weighted:', m-1)

end



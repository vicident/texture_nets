Learn neural network from one image! 

# Pretrained generators
You can find two iTorch notebooks as well as 8 pretrained models in `supplementary` directory.

# Train texture generator

# Prereq
torch
torch.cudnn
[display](https://github.com/szym/display)

# Train

This command should train a generator close to what is presented in the paper. The variance 
```
th texture_train.lua -texture data/textures/pebble.png -model_name pyramid -backend nn -num_iterations 1500 -vgg_no_pad -normalize_gradients
```

You may also explore other models. We found `pyramid2` requires bigger `learning rate` of about `5e-1`. To prevent degrading noise dimentionality should be increased: `noise_depth 16`. It also converges slower.

This works good for me: 
```
th texture_train.lua -texture data/textures/red-peppers256.o.jpg -gpu 0 -model_name pyramid2 -backend cudnn -num_iterations 1500 -vgg_no_pad -normalize_gradients -learning_rate 5e-1 -noise_depth 16
```

# Sample
For some reason nn.SpatialBatchNormalization throws a error in CPU mode, only GPU mode works for now. 

#

The code was tested with 12Gb Nvidia Tesla K40m GPU and Ubuntu 14.04. 


The code is based on 
https://github.com/jcjohnson/neural-style
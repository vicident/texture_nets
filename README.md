Learn neural network from one image!

## Pretrained models
You can find two `iTorch` notebooks as well as 8 pretrained models in `supplementary` directory. You need a GPU (nn.SpatialBatchNormalization throws a error in CPU mode), `torch`, and `iTorch` installed to try them.

## Train texture generator

### Prerequisites
- torch
- cunn
- torch.cudnn (optionally)
- [display](https://github.com/szym/display) (optionally)

Download VGG-19.
```
cd data/pretrained && bash download_models.sh && cd ../..
```

### Train

This command should train a generator close to what is presented in the paper. It is tricky, the variance in the results is rather high, many things lead to degrading (even optimizing for too long time). 
```
th texture_train.lua -texture data/textures/red-peppers256.o.jpg -model_name pyramid -backend nn -num_iterations 1500 -vgg_no_pad -normalize_gradients
```
Here is a sample of size `512x512` after learning for 700 iterations:
![GitHub Logo](data/sample.png)


You may also explore other models. We found `pyramid2` requires bigger `learning rate` of about `5e-1`. To prevent degrading noise dimentionality should be increased: `noise_depth 16`. It also converges slower.

This works good for me: 
```
th texture_train.lua -texture data/textures/red-peppers256.o.jpg -gpu 0 -model_name pyramid2 -backend cudnn -num_iterations 1500 -vgg_no_pad -normalize_gradients -learning_rate 5e-1 -noise_depth 16
```

- `vgg_no_pad` correspondes to padding option used in VGG. If set, padding mode = `valid`.

The samples and loss plot will apear at `display` web interface. 

### Sample

A sample from above can be obtained with
```
th texture_sample.lua -model data/out/model.t7 -noise_depth 3 -sample_size 512
```
`noise_depth` should correspond to `noise_depth` used when training.

## Stylization

The code will be uploaded shortly.

### Prepare

Extract content from `relu4_2` layer. We used Imagenet validation set.
```
th scripts/extract4_2.lua -gpu 1 -images_path path/to/image/dir
```


#
Was tested with 12Gb Nvidia Tesla K40m GPU and Ubuntu 14.04. 


The code is based on [Justin Johnson's code](https://github.com/jcjohnson/neural-style) for artistic style. 
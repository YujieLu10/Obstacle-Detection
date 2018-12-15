### Deformable Convolutional Networks

#### 1. Introduction

#### 2. Deformable Convolutional Networks

#### 3. Understanding Deformable ConvNets

> idea : augmenting the spatial sampling locations in convolution and RoI pooling with additional offsets and learning the offsets from target tasks

##### 3.1. In Context of Relate Works

**Spatial Transform Networks (STN)**

warps the faeture map via a global parametric transformation such as affine transformation
- warping is expensive
- learning the transformation parameters is difficult
- good in small scale image classification problem

*C.-H. Lin and S. Lucey. Inverse compositional spatial trans- former networks. arXiv preprint arXiv:1612.03897, 2016.* replaces the expensive feature warping by efficient tranformation parameter propagation

> Deformable convolution does not adopt a global parametric tranformation and feature warping. Instead, it samples the feature map in a local and dense manner.

**Active Convolution**

- also augments the sampling locations in the convolution with offsets
- learns the offsets via back-propagation end-to-end
- effective on image classification tasks

crutial differences from deformable convolution (less general and adaptive)
- shares the offsets all over the different spatial locations
- the offsets are *static model parameters* that are learnt per task or per training

deformable convolution
- offsets are *dynamic model outputs* vary per image location

**Effective Receptive Field**

pixels(in a receptive field) near the center have much larger impact on an output response

the effective receptive field only occupies a small fraction of the theoretical receptive field and has a Gaussian distribution

effetive receptive field size increases linearly with the *square root* of the number of convolutional layers

> indicates that even the top layers's unit in deep CNNs may not have large enough receptive field

deformable convolution
- learning receptive fields adaptively

**Atrous convolution** increases a normal filter's stride to be larger than 1 and keeps the original weights at sparsified sampling locations.

**Deformable Part Models(DPM)**
Deformable RoI pooling is similar to it : both methods learn the spatial deformation of object parts to maximize the classification score

DPM is not end-to-end and involves heuristic choices such as selection of components and part sizes

devormable ConvNets are deep and perform end-to-end training

**DeepID-Net**

similar spirit; complex; not clear how to adapt it to the recent state-of-the-art object detection methods in an end-to-end manner

**Spatial manipulation in RoI pooling**
Spatial pyramid pooling uses hand crafted pooling regions over scales.

**Transformation invariant features and their learning**

SIFT, CNNs cannot handle unknown transformations in the new tasks

> Our deformable modules generalize vatious transformations. The transformation invariance is learned from the target task.

#### Experiments


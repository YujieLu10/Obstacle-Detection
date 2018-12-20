### PVANet:Lightweight Deep Neural Networks for Real-time Object Detection

#### Abstract

A novel network structure, which is an oder of magnitude lighter than other state-of-the-art networks while maintaining the accuracy.

#### 1 Introduction

reduce the computational cost in the network design stage is still important

basic principle "smaller number of output channels with more layers"

#### 2 PVANet
##### 2.1 Feature extraction network

- Modified C.ReLU

- Inception structure

- Deep network training

adopt **residual structure** with **pre-activation** and **batch normalization**

our own policy : control the learning rate dynamically based on **plateau detection**. if the minimum loss is not updated for a certain number of iterations -> on-plateau -> the learning rate is decreased by a constant factor

- Overall design

##### 2.2 Object detection network

- Hyper-feature concatenation

> combining fine-grained details with highly abstracted information in the feature extraction layer helps the following region proposal network and classification network detect objects of different scales

the direct concatenation of all abstraction layers may produce redundant information with much higher compute requirement -> design the umber of different abstraction layers and the layer numbers of abstraction carefully

- Towards a more efficient detection network

> we found that feature inputs to the RPN does not need to be as deep as the inputs to the fully connected classifiers


#### 3 Experimental results
##### 3.1 ImageNet Pre-training
##### 3.2 VOC2007 detection
##### 3.3 VOC2012 detection

#### 4 Conclusions
> design a thin and light network which is capable of complex vision tasks
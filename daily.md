1. 配置caffe-priv环境，训练2D障碍物检测模型，基于obstacle2d_8cls_train0716_sub3数据集200000iteration的caffe model训练完成，测试训练的caffe model，熟悉工程。
2. survey论文，Deep Learning for Generic Object Detection了解目标检测的近期成果，阅读SNIPER论文，熟悉SNIPER工程，复现https://github.com/mahyarnajibi/SNIPER
3. 在我们的dataset上使用SNIPER，首先制作了 convert2coco_train.py 与 conver2coco_val.py 将train0829.txt与val0629.txt文件列表中的数据格式转换为coco的标准数据格式，存入data/coco/annotations中的instances_train.json中
4. 添加读取我们的训练集与验证集的方式于coco.py中，制作imdb与roidb。修改sniper_res101_e2e.yml中的process与thread参数，修改MNIteratorE2E中部分函数，正常创建processlist后，加载模型，初始化模型，在resnet_mx_101预训练模型基础上开始训练，遇到GPU内存不足问题，改小batch_size训练

1. 优化obstacle2d对小物体的检测，如远处的行人，自行车等
2. lib/dataset/coco.py 适应obastacle2d数据集存储方式
3. lib/data_utils/loaddata.py roidb与imdb源码阅读，并修改读取方式，加载obstacle2d中的imagedataset
4. 使用少量数据集，修改配置文件，设置不同的rpn anchor loader，以及rcnn rois batch size，进行训练
5. 使用大量数据集，进行相同的修改过程，进行训练
6. 对生成的第七个Epoch后的模型，进行测试，结果不佳，大部分框在了背景上面
7. 使用官方的instances_train与instance_val的标注与数据集，重新训练模型，用第七个Epoch后的模型测试obstacle2d的数据，	近距离的物体以及遮挡的物体皆有较好的检测结果
8. 输入模型的数据有一定问题，生成的roidb与imdb可能不符合要求，完全修改成coco的数据存储方式，重新训练模型

1. 解决SNIPER在train0829.txt数据训练下的检测结果为背景的问题，主要是segmentation的问题
2. 1/3原始图片size仍然取得较好的检测结果，检测速度与训练速度正常
3. 学习caffe与mxnet的网络结构，学习caffe的prototxt文件转换为mxnet的symbol.json的方式
4. SNIPER以train_ohem的结构训练，编写train_ohem.prototxt转换为mxnet网络结构
5. SNIPER use_ohem设为true，训练模型
6. 可视化train_ohem.prototxt，熟悉此网络结构
7. pvalite_mx_b5.py工具编写

1. 完成pvalite_b5.py，实现caffe的train.prototxt到mxnet的symbol.json的转换
2. 完成pvalite_b5.yml的配置文件编写
3. SNIPER使用train.prototxt的模型训练
4. 项目相关基础知识的学习补充
5. MNIterator.py修改部分的coding完成
6. pvalite_b5.py在SNIPER项目中，解决部分concat的bug
7. 逐层检查pvalite_b5的网络，检查从哪一层开始出现shape不匹配的问题

1. 修改网络结构，在网络的单层convolution的一路结构中，增加pooling层，shape对应
2. concat的bug解决，incleft,incmiddle,incright实现concat操作;conv3_down2 inc_concat inc3e实现concat操作
3. 逐层检查pvalite_b5的网络，解决shape不匹配问题
4. num classes，num hidden， dataset name等参数调整以匹配pvalite读取的数据集
5. init_weight_rcnn 与 init_weight_rpn 函数完成
6. 实现pvalite的checkpoint

1. 重新制作的无冗余的26个class的数据集，在resnet_101网络中跑出结果，验证为结果良好
2. 将按照train.prototxt编写的pvalite_b5.py的共享卷积层部分替换到resent_101网络上层的共享卷积层，RCNNLogLoss与RCNNAcc数值较为正常，等待SNIPER.params进行验证demo.py
3. 修改shape维度debug，比如代表object与非object两类
4. 测试卷积层模块与全连接层模块，分别与resnet网络拼接训练
5. 全连接层模块调通，与resnet拼接记录训练结果
6. 卷积层模块通过加batchnorm，降低lr方法防止了rpnloss与rcnnlogloss出现nan的跑飞情况
7. 训练模型测试基本通过，待精确计算map等指标
8. add bn并reduce lr后的训练结果，测试第七个epoch的模型，完成测试代码修改部分的coding，测试pvalite_b5的模型

1. add bn并reduce lr后的训练结果，测试最终epoch的模型。完成multiple images的测试代码。rpn结果可视化。
2. 借助coco.py与cocoeval.py等相关工具，编写testnet.py计算mAP（卡bug, cocoGT与cocoDT, dts为空）。发现main_test.py跑出的结果detections.json中的score过低。进行可视化，demo.py的调用aggregate后的visualize_dets结果正常，但是main_test.py 调用 get_detections.py 的visualize_dets结果不正常，框框位置不对，且score低。检查test数据集，未发现问题。修改coco.py，尝试把evaluate部分的infostr打印出来。addbn lr0.003的结果较之前更为低的lr的结果模型准确率较高一些 0.94 > 0.91 的样子，有待进一步优化网络结构与参数。
3. 计算mAP，测试结果的score过低问题解决。
4. 随即初始化以及选取迭代次数较少的第一个epoch训练结果模型作为初始化模型的实验结果比较。
5. 网络特性测试（先完成main_test部分）：person与person group区分；小尺度物体测试；类间差别小的物体测试，如面包车汽车卡车的识别结果；障碍物，车道block的检测稳定性。
6. deformable相关论文，微调网络结构，优化mAP，整理代码，新建工程SNIPER，提交至gitlab。
7. SNIPER只使用一个scale，看是否能有一样稳定的检测结果。不同scales与valid的ranges测试resnet模型与pvalite模型。此前训练输入为[400,640]resize后的数据集，重新制作原尺寸的images与annotations，并push实验记录。
8. 原尺寸训练resnet模型，以及pvalite模型。



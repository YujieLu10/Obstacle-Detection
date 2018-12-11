12.5
重新制作的无冗余的26个class的数据集，在resnet_101网络中跑出结果，验证为结果良好
12.6
将按照train.prototxt编写的pvalite_b5.py的共享卷积层部分替换到resent_101网络上层的共享卷积层，RCNNLogLoss与RCNNAcc数值较为正常，等待SNIPER.params进行验证demo.py
修改shape维度(0, 8,  -1, 0) -> (0, 2,  -1, 0) 2应该代表object与非object两类
12.7
      1. 测试卷积层模块与全连接层模块
      2. 全连接层模块调通
12.8
      1. 卷积层模块通过加batchnorm，降低lr方法防止了rpnloss与rcnnlogloss出现nan的跑飞情况
      2. 训练模型测试基本通过，待精确计算mAP等指标

12.9
      1. add bn并reduce lr后的训练结果，测试第七个epoch的模型
      2. 调通main_test.py

12.10
      1. 借助coco.py与cocoeval.py等相关工具，编写testnet.py计算mAP（卡bug, cocoGT与cocoDT, dts为空）
      2. 发现main_test.py跑出的结果detections.json中的score过低
      3. 进行可视化，demo.py的调用aggregate后的visualize_dets结果正常，但是main_test.py 调用 get_detections.py 的visualize_dets结果不正常，框框位置不对，且score低
      4. 检查test数据集，未发现问题
      5. 修改coco.py，尝试把evaluate部分的infostr打印出来
      6. addbn lr0.003的结果较之前更为低的lr的结果模型准确率较高一些 0.94 > 0.91 的样子，有待进一步优化网络结构与参数

12.11
- [ ]  计算mAP
- [ ] main_test结果的score问题
    - 网络特性测试
    - person与person group区分
    - 小尺度物体测试
    - 类间差别小的物体测试，如面包车汽车卡车的识别结果
    - 障碍物，车道block的检测稳定性

 - [ ] rpn结果可视化
 - [ ] 看deformable相关论文
 - [ ] 微调网络结构，优化mAP


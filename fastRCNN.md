参考原文：https://blog.csdn.net/u014380165/article/details/72851319

R-CNN问题
训练分多步
fine tuning一个预训练的网络
针对每个类别都训练一个SVM分类器
利用regressors对bounding-box进行回归
（region proposal需单独用selective search方式获得）
时间和内存消耗比较大
在训练SVM和回归的时候需要用网络训练的特征作为输入
特征保存在磁盘上再读入的时间消耗还是比较大的
测试时比较慢，每张图片的每个region proposal都要做卷机，重复操作太多

SPPnet算法解决RCNN中重复卷机问题，缺点如下
训练步骤过多
需要训练SVM分类器
需要额外的回归器
特征也是保存在磁盘上
Fast RCNN，不仅训练步骤减少，也不需要额外奖特征保存在磁盘上


Apollo 将ROI过滤器应用于点云和图像数据 -> 缩小范围加快感知

通过监测网络馈送以过滤的点云
输出用语构建围绕对象的三位框架

Point Cloud -> Detection Network -> Center Offset; Objectness; Positiveness; Object Height; Class Probability
Which lights pertain to the lane?


### 传感数据比较

### 感知融合策略
激光雷达和雷达检测障碍物
融合输出的主要算法：卡尔曼滤波（预测和更新的无限循环）
Predict State
Use information we have to predict the state
Update Measurement
use new observations to correct our belief

异步融合
逐个更新所收到的传感器测量结果
同步融合
同时更新来自不同传感器的测量结果
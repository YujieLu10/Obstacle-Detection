Levels of Driving Automation
Level1 Driver Assistance Driver Fully Engaged
Level2 Partial Automation Automatic Cruise Control Automatic Lane Keeping
Level3 Conditional Automation Human Take Over Whenever Necessary
Level4 No Human Interference Without Steering Wheel, Throttle or Brake Restricted in Geofence
Level5 Full Automation


线控驾驶车辆
控制器区域网络（CAN）使车辆的内部通信网络
计算机系统通过CAN卡连接汽车内部网络，发送加速，制动和转向信号
 
全球定位系统（GPS）通过绕地卫星接收信号确定我们的位置 
惯性测量装置（IMU） 测量车辆的运动和位置（跟踪位置，速度，加速度和其他）
激光雷达（LiDAE）由一组脉冲激光器组成
 Apollo使用的激光雷达可360度扫描车辆周围，这些激光束的反射形成了软件可用于了解环境的点云

摄像头捕获图像数据，计算机视觉提取图像内容了解环境

雷达检测障碍物，分辨率低，难以分辨是何种障碍物
雷达优点：适用于各种天气和照明条件，特别擅长其他车辆的速度

整体框架 开放式软件层
实时操作系统 RTOS
确保给定时间内完成任务
加入Apollo设计的内核，ubuntu成为一个RTOS
运行时框架
ROS定制版
Apollo RTOS上运行的软件框架
ROS根据功能将自制系统划分为多个模块
模块互相独立，运行时相互通信
改进共享内存的功能和性能，去中心化（域中的每个节点都有关于域中其他节点的信息，公共域取代原来的ROS主节点解决单点故障问题）和数据兼容性（ROS Message，一个节点更新后通信时与其他节点消息格式不一致导致通信失败，严重的兼容性问题，改进为 protobuf，一种结构化数据序列化方法这对开发用于通过电线彼此通信或用于存储数据的程序非常有用，将新字段添加到消息格式中，而不会破坏向后兼容性）
应用程序模块层
MAP Engine
Localization
Perception
Planning
Control
End-to-End
Human Machine Interface

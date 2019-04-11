2019/1/3

main_test & demo 测试在1920 1200尺寸上训练的pvalite模型

- 线上multi forward不现实，考虑single forward
- 关注下模型在block（路障）上的性能怎么样，现在线上的版本不是很稳定

测试pvalite orisize训练的模型在单一scale下的效果(single forward)

- 换用train1030（较多市区数据）

1/4

main_test pvalite 单一尺度

单一尺度pvalite 与线上模型（/private/ningqingqun/caffemodels/liteb5/）量化对比 

JPGImages图片缺失
> luyujie@gpu13:~/SNIPER$ python demo.py

./output/pvalite_b5_bn/pvalite_b5/train2014_val2014/SNIPER-0007.params
Traceback (most recent call last):
  File "demo.py", line 123, in <module>
    main()
  File "demo.py", line 99, in main
    mod.init_params(arg_params=arg_params, aux_params=aux_params)
  File "SNIPER-mxnet/python/mxnet/module/module.py", line 309, in init_params
  File "SNIPER-mxnet/python/mxnet/module/module.py", line 297, in _impl
  File "SNIPER-mxnet/python/mxnet/ndarray/ndarray.py", line 1970, in copyto
  File "<string>", line 25, in _copyto
  File "SNIPER-mxnet/python/mxnet/_ctypes/ndarray.py", line 92, in _imperative_invoke
  File "SNIPER-mxnet/python/mxnet/base.py", line 149, in check_call
mxnet.base.MXNetError: [14:59:57] src/operator/nn/./../tensor/../elemwise_op_common.h:123: Check failed: assign(&dattr, (*vec)[i]) Incompatible attr in node  at 0-th output: expected [27], got [20]

Stack trace returned 10 entries:
[bt] (0) /home/luyujie/SNIPER/SNIPER-mxnet/python/mxnet/../../lib/libmxnet.so(dmlc::StackTrace[abi:cxx11]()+0x5b) [0x7fb67da9b8ab]
[bt] (1) /home/luyujie/SNIPER/SNIPER-mxnet/python/mxnet/../../lib/libmxnet.so(dmlc::LogMessageFatal::~LogMessageFatal()+0x28) [0x7fb67da9c418]
[bt] (2) /home/luyujie/SNIPER/SNIPER-mxnet/python/mxnet/../../lib/libmxnet.so(bool mxnet::op::ElemwiseAttr<nnvm::TShape, &mxnet::op::shape_is_none, &mxnet::op::shape_assign, true, &mxnet::op::shape_string[abi:cxx11], -1, -1>(nnvm::NodeAttrs const&, std::vector<nnvm::TShape, std::allocator<nnvm::TShape> >*, std::vector<nnvm::TShape, std::allocator<nnvm::TShape> >*, nnvm::TShape const&)::{lambda(std::vector<nnvm::TShape, std::allocator<nnvm::TShape> >*, unsigned long, char const*)#1}::operator()(std::vector<nnvm::TShape, std::allocator<nnvm::TShape> >*, unsigned long, char const*) const+0xbf1) [0x7fb67dacdd71]
[bt] (3) /home/luyujie/SNIPER/SNIPER-mxnet/python/mxnet/../../lib/libmxnet.so(bool mxnet::op::ElemwiseShape<1, 1>(nnvm::NodeAttrs const&, std::vector<nnvm::TShape, std::allocator<nnvm::TShape> >*, std::vector<nnvm::TShape, std::allocator<nnvm::TShape> >*)+0x24a) [0x7fb67dacfc2a]
[bt] (4) /home/luyujie/SNIPER/SNIPER-mxnet/python/mxnet/../../lib/libmxnet.so(mxnet::imperative::SetShapeType(mxnet::Context const&, nnvm::NodeAttrs const&, std::vector<mxnet::NDArray*, std::allocator<mxnet::NDArray*> > const&, std::vector<mxnet::NDArray*, std::allocator<mxnet::NDArray*> > const&, mxnet::DispatchMode*)+0xe11) [0x7fb680031991]
[bt] (5) /home/luyujie/SNIPER/SNIPER-mxnet/python/mxnet/../../lib/libmxnet.so(mxnet::Imperative::Invoke(mxnet::Context const&, nnvm::NodeAttrs const&, std::vector<mxnet::NDArray*, std::allocator<mxnet::NDArray*> > const&, std::vector<mxnet::NDArray*, std::allocator<mxnet::NDArray*> > const&)+0x35f) [0x7fb6800189af]
[bt] (6) /home/luyujie/SNIPER/SNIPER-mxnet/python/mxnet/../../lib/libmxnet.so(MXImperativeInvokeImpl(void*, int, void**, int*, void***, int, char const**, char const**)+0x9cf) [0x7fb6804c4e9f]
[bt] (7) /home/luyujie/SNIPER/SNIPER-mxnet/python/mxnet/../../lib/libmxnet.so(MXImperativeInvokeEx+0x40b) [0x7fb6804c600b]
[bt] (8) /usr/lib/x86_64-linux-gnu/libffi.so.6(ffi_call_unix64+0x4c) [0x7fb6919f1e40]
[bt] (9) /usr/lib/x86_64-linux-gnu/libffi.so.6(ffi_call+0x2eb) [0x7fb6919f18ab]

class_name 和 num_class 不一致，训练时27，测试的数据20个class

解决方法：pvalite_b5.yml中numclasses设为27，pvalite_b5.py中num_class设为27，convert2coco_test.py的class数量设为27
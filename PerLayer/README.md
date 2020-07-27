
## TVM 开启DebugDumpGroup
Path: tvm0.6/src/relay/pass/fuse_op.cc or tvm0.7/src/relay/transforms/fuse_ops.cc
将fuse_op.cc中的下方代码:
```
// this->DebugDumpGroup(body);
```
改为:
```
if(fuse_opt_level>0){
  std::cout<< "-->Begin to DebugDumpGroup<--"<<std::endl;
  this->DebugDumpGroup(body);
  std::cout<< std::endl << "-->End of DebugDumpGroup<--"<<std::endl;
}
```
将fuse_op.cc中的下方代码:
```
LOG(INFO) << "Dump of group info:\n" << text;
```
改为:
```
std::cout << "Dump of group info:\n" << text;
```
重新build tvm，然后编译模型（例如：onnx模型），将编译过程输出到./output 文件

## 处理DebugDumpGroup数据
1. python3 process.py --output ./output 
2. 首先输出每个node的信息（node id，node op, node input，node group）属于同一个group的node会被fuse到一起
3. 然后输出Dominator Node，同一group的node会被fuse到dominator node
4. 最后输出fused_op，即fused per layer



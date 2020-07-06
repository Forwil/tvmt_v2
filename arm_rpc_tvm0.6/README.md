TVM-0.7 对ARM设备支持不完善，目前使用TVM-0.6版本

功能：现在支持直接从ONNX获取模型、编译、RPC通信和测速。
      将onnx模型转成relay模型，当输入log时候，则pick 最好的log，可以指定batch size

使用：分别在HOST端和DEVICE端配置RPC环境（详见rpc_tools）

样例1（不使用log进行转换）：python3 speed.py xxx.onnx -d aarch64

样例2（使用log进行转换）：python3 speed.py xxx.onnx -d aarch64 -l xxx.log

python3 speed.py --help可输出帮助信息

输出：\[模型名字\] , \[耗时(ms)\]

tune.py

功能：对onnx模型进行autotune，可以指定device，tune的轮数，以及使用的batch size

样例1：python3 tune.py xxx.onnx -d aarch64 -t 1000 -b 1 -f rasp3b

输出：在logs目录下生成log，第二次启动会从logs中获取最好的参数进行继续训练


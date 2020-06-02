# tvmt_v2

## convert.py

功能：将onnx模型转成relay模型，当输入log时候，则pick 最好的log，可以指定batch size

样例1（不使用log进行转换）：python tvmt/convert.py xxx.onnx -d gpu
样例2（使用log进行转换）：python tvmt/convert.py xxx.onnx -d gpu -l xxx.log
默认在relay_model下生成名字对应的tar/params/json，可以使用-o指定输出路径
python tvmt/convert.py --help可输出帮助信息

## speed.py

功能：执行relay模型（暂不支持remote模式），进行测速

样例1（不使用log进行转换）：python tvmt/speed.py relay_model/resnet18.onnx
输出：\[模型名字\] , \[耗时(ms)\]

## tune.py

功能：对onnx模型进行autotune，可以指定device，tune的轮数，以及使用的batch size

样例1（不使用log进行转换）：python tvmt/tune.py xxx.onnx -d gpu -t 1000 -b 16
输出：在logs目录下生成log，第二次启动会从logs中获取最好的参数进行继续训练



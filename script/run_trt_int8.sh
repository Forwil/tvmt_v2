for i in `cat list`
do
    echo $i >> trt_b1_int8.log
    giexec --onnx=models/$i.onnx --device=0 --batch=1 --int8 | grep mean >> trt_b1_int8.log
done

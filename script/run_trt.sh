for i in `cat list`
do
    echo $i >> trt_b1.log
    giexec --onnx=models/$i.onnx --device=0 --batch=1 | grep mean >> trt_b1.log
done

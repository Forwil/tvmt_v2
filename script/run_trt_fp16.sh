for i in `cat list`
do
    echo $i >> trt_b1_fp16.log
    giexec --onnx=models/$i.onnx --device=0 --batch=1 --fp16 | grep mean >> trt_b1_fp16.log
done

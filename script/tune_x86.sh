for i in `cat list`
do
    python tune.py models/$i.onnx -d x86 -t 100
done

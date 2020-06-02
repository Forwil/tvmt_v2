for i in `cat list`
do
    python convert.py models/$i.onnx -d x86 -o x86/$i
done

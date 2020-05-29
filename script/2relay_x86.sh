for i in `cat list`
do
    python onnx_to_relay.py models/$i.onnx -d x86 -o x86/$i
done

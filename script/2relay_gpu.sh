for i in `cat list`
do
    python onnx2relay.py models/$i.onnx -d gpu -o gpu/$i
done

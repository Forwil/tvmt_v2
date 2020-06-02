for i in `cat list`
do
    python convert.py models/$i.onnx -d gpu -o gpu/$i
done

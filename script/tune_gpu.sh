for i in `cat list`
do
    python tune.py models/$i.onnx -d gpu -t 100
done

for i in `cat list`
do
    python tvmt/tune.py models/$i.onnx -d gpu -t 100 -f t4
done

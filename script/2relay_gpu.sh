for i in `cat list`
do
    python tvmt/convert.py models/$i.onnx -d gpu 
done

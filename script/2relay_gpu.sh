for i in `cat list`
do
    python onnx_to_relay.py models/$i.onnx -d gpu -o gpu/$i
done

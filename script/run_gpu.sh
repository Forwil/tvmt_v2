for i in `cat list`
do
    python tvmt/speed.py relay_model/${i}.onnx_gpu_b1 -d gpu | tee gpu_2080ti.log
done


export PYTHONPATH=path_to_tvm/tvm/python
export TVM_NUM_THREADS=1
python -m tvm.exec.rpc_server --tracker=[HOST_IP]:9190 --key=pi


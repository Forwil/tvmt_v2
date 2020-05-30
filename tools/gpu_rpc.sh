# python3 -m tvm.exec.rpc_tracker --host=0.0.0.0 --port=9190 
# python3 -m tvm.exec.rpc_server --tracker=0.0.0.0:9190 --key=V100
# python3 -m tvm.exec.query_rpc_tracker --port 9190
python3 -m tvm.exec.rpc_tracker --host=0.0.0.0 --port=9190 &
CUDA_VISIBLE_DEVICES=0 python3 -m tvm.exec.rpc_server --tracker=0.0.0.0:9190 --key=t4 &
CUDA_VISIBLE_DEVICES=1 python3 -m tvm.exec.rpc_server --tracker=0.0.0.0:9190 --key=t4 &
CUDA_VISIBLE_DEVICES=2 python3 -m tvm.exec.rpc_server --tracker=0.0.0.0:9190 --key=t4 &
CUDA_VISIBLE_DEVICES=3 python3 -m tvm.exec.rpc_server --tracker=0.0.0.0:9190 --key=t4 &
CUDA_VISIBLE_DEVICES=4 python3 -m tvm.exec.rpc_server --tracker=0.0.0.0:9190 --key=t4 &
CUDA_VISIBLE_DEVICES=5 python3 -m tvm.exec.rpc_server --tracker=0.0.0.0:9190 --key=t4 &
CUDA_VISIBLE_DEVICES=6 python3 -m tvm.exec.rpc_server --tracker=0.0.0.0:9190 --key=t4 &
CUDA_VISIBLE_DEVICES=7 python3 -m tvm.exec.rpc_server --tracker=0.0.0.0:9190 --key=t4 &

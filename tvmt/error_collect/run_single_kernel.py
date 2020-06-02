import tvm
import numpy as np

import topi.cuda.conv2d_int8

def get_runtime_array(args, ctx=None):
    res = []
    for item in args:
        shape = [x.value for x in item.shape]
        arr = tvm.nd.array(np.zeros(shape, dtype=item.dtype), ctx=ctx)
        res.append(arr)
    return tuple(res)

def main():
    i = tvm.te.placeholder([8, 4, 128, 128], dtype='int8', name='name')
    k = tvm.te.placeholder([4, 4, 64, 64], dtype='int8', name='name')
    t = tvm.autotvm.task.create(
        'conv2d_NCHWc_int8.cuda', (i, k, 1, 0, 1, 'NCHW', 'int32'), target='cuda')
    # 18910940 of task conv2d_NCHWc_int8.cuda
    # results in cudaError: CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES
    idx = 18910940
    # idx = 0
    with tvm.target.cuda():
        s, args = t.instantiate(t.config_space.get(idx))
        ctx = tvm.context('gpu')
        tensors = get_runtime_array(args, ctx)
        func = tvm.build(s, args, target='cuda')
        func(*tensors)

if __name__ == '__main__':
    main()
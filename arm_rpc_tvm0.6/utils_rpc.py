from tvm.contrib.util import tempdir
import sys
sys.path.append('../tvmt/')
from utils import *

def create_ctx_rpc(device, lib, did = 0):
    if device == "aarch64" or device == 'arm':
        from tvm import autotvm
        device_key = 'pi'
        remote = autotvm.measure.request_remote(device_key, '0.0.0.0', 9190,
                                                timeout=10000)
        # export library
        tmp = tempdir()
        filename = "net.tar"
        lib.export_library(tmp.relpath(filename))

        remote.upload(tmp.relpath(filename))
        rlib = remote.load_module(filename)

        # upload parameters to device
        target = create_target(device)
        ctx = remote.context(str(target), did)
        return ctx, rlib
    else:
        return None, None


def speed_rpc(graph, lib, params, ctx):
    import numpy as np
    import tvm.contrib.graph_runtime as runtime
    import json
    graph_dict = json.loads(graph)
    input_shape = graph_dict["attrs"]["shape"][1][0]
    input_name = graph_dict["nodes"][0]["name"]
    data_tvm = tvm.nd.array(np.random.uniform(size = input_shape).astype("float32"))
    module = runtime.create(graph, lib, ctx)
    module.set_input(input_name, data_tvm)

    #module.load_params(params)
    module.set_input(**params)

    #ftimer = module.module.time_evaluator("run", ctx, number = 1, repeat = 100)
    ftimer = module.module.time_evaluator("run", ctx, number = 1, repeat = 10)
    prof_res = np.array(ftimer().results) * 1000
    return np.mean(prof_res)


def speed_rpc_profile(graph, lib, params, ctx):
    import numpy as np
    #import tvm.contrib.graph_runtime as runtime
    import json
    graph_dict = json.loads(graph)
    input_shape = graph_dict["attrs"]["shape"][1][0]
    input_name = graph_dict["nodes"][0]["name"]
    data_tvm = tvm.nd.array(np.random.uniform(size = input_shape).astype("float32"))
    from tvm.contrib.debugger import debug_runtime as runtime
    print("use debug_runtime")
    module = runtime.create(graph, lib, ctx)
    module.set_input(input_name, data_tvm)

    #module.load_params(params)
    module.set_input(**params)

    #ftimer = module.module.time_evaluator("run", ctx, number = 1, repeat = 100)
    ftimer = module.module.time_evaluator("run", ctx, number = 1, repeat = 10)
    prof_res = np.array(ftimer().results) * 1000
    return np.mean(prof_res)



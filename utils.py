import tvm.relay as relay
import tvm

def create_target(device):
    if device == "x86":
        target = tvm.target.create("llvm")
    elif device == "gpu":
        target = tvm.target.cuda()
    return target

def create_ctx(device, did = 0):
    if device == "x86":
        ctx = tvm.cpu(did)
    elif device == "gpu":
        ctx = tvm.gpu(did)
    return ctx

def speed(graph, lib, params, ctx):
    import numpy as np
    import tvm.contrib.graph_runtime as runtime
    import json
    graph_dict = json.loads(graph)
    input_shape = graph_dict["attrs"]["shape"][1][0]
    input_name = graph_dict["nodes"][0]["name"]
    data_tvm = tvm.nd.array(np.random.uniform(size = input_shape).astype("float32"))
    module = runtime.create(graph, lib, ctx)
    module.set_input(input_name, data_tvm)
    module.load_params(params)
    ftimer = module.module.time_evaluator("run", ctx, number = 1, repeat = 100)
    prof_res = np.array(ftimer().results) * 1000
    return np.mean(prof_res)

def get_onnx(path):
    import onnx
    on = onnx.load(open(path, "rb"))
    name = on.graph.input[0].name
    input_shape = [i.dim_value for i in  on.graph.input[0].type.tensor_type.shape.dim]  
    return on, {name : input_shape}

def get_model(path):
    graph = open(path + ".json").read()
    lib = tvm.module.load(path + ".tar")
    params = bytearray(open(path + ".params", "rb").read())
    return graph, lib, params

def build_model_from_onnx(onnx_model, input_shape, target, log = ""):
    from tvm import autotvm
    import os
    model, relay_params = relay.frontend.from_onnx(onnx_model, input_shape)
    func = model["main"]
    if os.path.isfile(log):
        with autotvm.apply_history_best(log):
            with relay.build_config(opt_level=3):
                graph, lib, params = relay.build(func , target, params = relay_params)
    else:
        with relay.build_config(opt_level=3):
            graph, lib, params = relay.build(func , target, params = relay_params)

        
    return graph, lib , params

def save_model(graph, lib, params, prefix = "relay"):
    deploy_name = prefix
    lib.export_library(deploy_name + '.tar' )
    with open(deploy_name + ".json", "w") as fo:
        fo.write(graph)
    with open(deploy_name + ".params", "wb") as fo:
        fo.write(relay.save_param_dict(params))
    return True

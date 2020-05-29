import tvm

device = "x86"

def get_model(path):
    graph = open(path + ".json").read()
    lib = tvm.module.load(path + ".tar")
    params = bytearray(open(path + ".params", "rb").read())
    return graph, lib, params

def create_target(device, did = 1):
    if device == "x86":
        target = tvm.cpu(did)
    elif device == "gpu":
        target = tvm.gpu(did)
    return target

def speed(graph, lib, param, target):
    import numpy as np
    import tvm.contrib.graph_runtime as runtime
    import json
    graph_dict = json.loads(graph)
    input_shape = graph_dict["attrs"]["shape"][1][0]
    input_name = graph_dict["nodes"][0]["name"]
    data_tvm = tvm.nd.array(np.random.uniform(size = input_shape).astype("float32"))
    module = runtime.create(graph, lib, target)
    module.set_input(input_name, data_tvm)
    module.load_params(params)
    ftimer = module.module.time_evaluator("run", target, number = 1, repeat = 10)
    prof_res = np.array(ftimer().results) * 1000
    return np.mean(prof_res)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description = "run relay model")
    parser.add_argument("relay", help = "relay model path")
    parser.add_argument("-d", "--device", default="x86", choices=["gpu","x86"])
    arg = parser.parse_args() 
    graph, lib, params = get_model(arg.relay)
    target = create_target(arg.device) 
    time =speed(graph, lib, params, target)
    import os
    name = os.path.basename(arg.relay)
    print("%s, %.2f" % (name, time))

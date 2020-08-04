
from utils_rpc import *

if __name__ == "__main__":
    import argparse
    import os
    parser = argparse.ArgumentParser(description = "onnx to tvm relay model")
    parser.add_argument("onnx", help = "onnx model path")
    parser.add_argument("-l", "--log", default="", type = str)
    parser.add_argument("-d", "--device", default="x86", choices=["gpu","x86","arm","aarch64"])
    parser.add_argument("-o", "--output", default="", type = str)
    parser.add_argument("-b", "--batch", default=1, type = int)
    parser.add_argument("-p", "--profile", default="false", type = str)
    arg = parser.parse_args()
    on, input_shape = get_onnx(arg.onnx, arg.batch)
    target = create_target(arg.device)
    graph, lib, params = build_model_from_onnx(on, input_shape, target, log = arg.log)
    if arg.output == "":
        arg.output = "./relay_model/" + os.path.basename(arg.onnx) + "_" + arg.device + "_b" + str(arg.batch)
    if arg.log == "":
        arg.log = "notuned"
    arg.log = os.path.basename(arg.log)

    ctx, rlib = create_ctx_rpc(arg.device, lib) 

    if arg.device == 'aarch64' or arg.device == 'arm':
        lib = rlib
    if arg.profile == 'false':
        time = speed_rpc(graph, lib, params, ctx)
    elif arg.profile == 'true':
        time = speed_rpc_profile(graph, lib, params, ctx)
    name = os.path.basename(arg.onnx)
    print("%s, %.2f" % (name, time))


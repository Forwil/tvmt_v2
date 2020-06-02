from utils import *
if __name__ == "__main__":
    import argparse
    import os
    parser = argparse.ArgumentParser(description = "onnx to tvm relay model")
    parser.add_argument("onnx", help = "onnx model path")
    parser.add_argument("-l", "--log", default="", type = str)
    parser.add_argument("-d", "--device", default="x86", choices=["gpu","x86"])
    parser.add_argument("-o", "--output", default="", type = str)
    parser.add_argument("-b", "--batch", default=1, type = int)
    arg = parser.parse_args()
    on, input_shape = get_onnx(arg.onnx, arg.batch)
    target = create_target(arg.device)
    graph, lib, params = build_model_from_onnx(on, input_shape, target, log = arg.log)
    if arg.output == "":
        arg.output = "./relay_model/" + os.path.basename(arg.onnx) + "_" + arg.device + "_b" + str(arg.batch)
    if arg.log == "":
        arg.log = "_notuned"
    arg.log = os.path.basename(arg.log)
    save_model(graph, lib, params, arg.output + arg.log)

from utils import *
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description = "onnx to tvm relay model")
    parser.add_argument("onnx", help = "onnx model path")
    parser.add_argument("-l", "--log", default="", type = str)
    parser.add_argument("-d", "--device", default="x86", choices=["gpu","x86"])
    parser.add_argument("-o", "--output", default="relay_model", type = str)
    arg = parser.parse_args()
    on, input_shape = get_onnx(arg.onnx)
    target = create_target(arg.device)
    graph, lib, params = build_model_from_onnx(on, input_shape, target, log = arg.log)
    save_model(graph, lib, params, arg.output + arg.log)

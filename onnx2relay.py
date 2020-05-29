import tvm.relay as relay
import tvm

def get_onnx(path):
    import onnx
    on = onnx.load(open(path, "rb"))
    name = on.graph.input[0].name
    input_shape = [i.dim_value for i in  on.graph.input[0].type.tensor_type.shape.dim]  
    return on, {name : input_shape}

def create_target(device):
    if device == "x86":
        target = tvm.target.create("llvm")
    elif device == "gpu":
        target = tvm.target.cuda()
    return target

def build_model(onnx_model, input_shape, target):
    model, relay_params = relay.frontend.from_onnx(onnx_model, input_shape)
    func = model["main"]
    with relay.build_config(opt_level=3):
        graph, lib, params = relay.build(func , target, params = relay_params)
    return graph, lib , params

def save_model(graph, lib, params, prefix):
    deploy_name = prefix
    lib.export_library(deploy_name + '.tar' )
    with open(deploy_name + ".json", "w") as fo:
        fo.write(graph)
    with open(deploy_name + ".params", "wb") as fo:
        fo.write(relay.save_param_dict(params))
    return True

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description = "onnx to tvm relay model")
    parser.add_argument("onnx", help = "onnx model path")
    parser.add_argument("-d", "--device", default="x86", choices=["gpu","x86"])
    parser.add_argument("-o", "--output", default="relay_model", type = str)
    arg = parser.parse_args()
    on, input_shape = get_onnx(arg.onnx)
    target = create_target(arg.device)
    graph, lib, params = build_model(on, input_shape, target)
    save_model(graph, lib, params, arg.output)

from utils import *
from tvm import autotvm

def get_tasks_from_onnx(on, input_shape, target):
    from tvm import relay
    mod, params = relay.frontend.from_onnx(on, input_shape) 
    func = mod["main"] 
    ops = [
        relay.op.nn.conv2d,
        relay.nn.batch_matmul,
        relay.nn.dense,
        relay.nn.conv2d_transpose,
        ]
    tasks = autotvm.task.extract_from_program(func, target = target,
                                params = params,
                                ops = ops)
    return tasks

def create_measure(device):
    if device == 'arm' or device == 'aarch64':
        measure_option = autotvm.measure_option(
        builder=autotvm.LocalBuilder(
        build_func='ndk' if use_android else 'default'),
        runner=autotvm.RPCRunner(
        "pi", host='0.0.0.0', port=9190,
        number=5,
        timeout=10,
        ))
    elif device == 'x86':
        measure_option = autotvm.measure_option(
        builder=autotvm.LocalBuilder(),
        runner=autotvm.LocalRunner(number=5, repeat=1,
        min_repeat_ms=1000),
       )
    elif device == 'gpu':
        measure_option = autotvm.measure_option(
        builder=autotvm.LocalBuilder(timeout=10),
        runner=autotvm.RPCRunner(
        't4',  # change the device key to your key
        '0.0.0.0', 9190,
        number=20, repeat=3, timeout=4, min_repeat_ms=150)
        )
    return measure_option

def tune_task(tasks, measure, resume_log_file = "tune.log", n_trial = 10):
    from tvm.autotvm.tuner import XGBTuner
    import os
    for idx , task in enumerate(reversed(tasks)):
        prefix = "[Task %2d/%2d] " % (idx + 1, len(tasks) )
        tuner = XGBTuner(task, loss_type = 'rank')
        if os.path.isfile(resume_log_file):
            tuner.load_history(autotvm.record.load_from_file(resume_log_file)) 
        n_try = min(n_trial, len(task.config_space))
        tuner.tune(n_trial = n_try,
                    early_stopping = 80,
                    measure_option = measure,
                    callbacks = [
                        autotvm.callback.progress_bar(n_try, prefix = prefix),
                        autotvm.callback.log_to_file(resume_log_file)
                    ])
    return 

if __name__ == "__main__":
    import argparse
    import os
    parser = argparse.ArgumentParser(description = "tune from onnx relay model")
    parser.add_argument("onnx", help = "onnx model path")
    parser.add_argument("-d", "--device", default="x86", choices=["gpu","x86"])
    parser.add_argument("-t", "--time", default=10, type = int)
    arg = parser.parse_args()

    on, input_shape = get_onnx(arg.onnx)
    target = create_target(arg.device)
    measure = create_measure(arg.device)
    tasks = get_tasks_from_onnx(on, input_shape, target)
    print("Got %d task to tune" % (len(tasks)))
    name = os.path.basename(arg.onnx) + "_" + arg.device + ".log"
    tune_task(tasks, measure, resume_log_file = name, n_trial = arg.time)

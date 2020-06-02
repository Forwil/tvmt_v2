from utils import *
from tvm import autotvm

def get_tasks_from_onnx(on, input_shape, target):
    from tvm import relay
    mod, params = relay.frontend.from_onnx(on, input_shape) 
    func = mod["main"] 
    ops = [
        relay.op.get("nn.conv2d"),
        relay.op.get("nn.batch_matmul"),
        relay.op.get("nn.dense"),
        relay.op.get("nn.conv2d_transpose"),
        ]
    tasks = autotvm.task.extract_from_program(func, target = target,
                                params = params,
                                ops = ops)
    for i in range(len(tasks)):
        try:  
            tsk = autotvm.task.create(tasks[i].name, tasks[i].args,
                                      tasks[i].target, tasks[i].target_host, 'winograd')
            input_channel = tsk.workload[1][1]
            if input_channel >= 64:
                print(tasks[i].name + " goto winograd " + tsk.name , tasks[i].args)
                tasks[i] = tsk
        except Exception:
            pass

    return tasks

def create_measure(device, flag = "t4"):
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
 #       runner=autotvm.LocalRunner(number=5, repeat=1,
 #       min_repeat_ms=1000),
       runner=autotvm.RPCRunner(
        'x86',  # change the device key to your key
        '0.0.0.0', 9190,
        number=20, repeat=3, timeout=4, min_repeat_ms=150)

       )
    elif device == 'gpu':
        measure_option = autotvm.measure_option(
        builder=autotvm.LocalBuilder(timeout=100),
        runner=autotvm.RPCRunner(
        flag,  # change the device key to your key
        '0.0.0.0', 9190,
        number=20, repeat=3, timeout=100, min_repeat_ms=150)
        )
    return measure_option

def tune_task(name, tasks, measure, resume_log_file = "tune.log", n_trial = 10):
    from tvm.autotvm.tuner import XGBTuner
    import os
    dir_name = os.path.dirname(resume_log_file)
    try:
        os.mkdir(dir_name)
    except:
        pass
    for idx , task in enumerate(reversed(tasks)):
        prefix = "[%s][Task %2d/%2d] " % (name, idx + 1, len(tasks) )
        tuner = XGBTuner(task, loss_type = 'rank')
        if os.path.isfile(resume_log_file):
            print("load log file:" + resume_log_file)
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
    parser.add_argument("-b", "--batch", default=1, type = int)
    parser.add_argument("-f", "--flag", default="t4", type = str)
    arg = parser.parse_args()

    on, input_shape = get_onnx(arg.onnx,arg.batch)
    target = create_target(arg.device)
    measure = create_measure(arg.device, arg.flag)
    tasks = get_tasks_from_onnx(on, input_shape, target)
    print("Got %d task to tune" % (len(tasks)))
    for i in tasks:
        print(i.name, i.config_space)    
    name = os.path.join("logs", os.path.basename(arg.onnx) + "_" + arg.device + "_" + str(arg.batch) +".log")
    tune_task(arg.onnx, tasks, measure, resume_log_file = name, n_trial = arg.time)

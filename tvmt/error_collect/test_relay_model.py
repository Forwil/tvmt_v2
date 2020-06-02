import tvm
import tvm.relay
import tvm.relay.testing
import tvm.autotvm


def tune_tasks(tasks,
               measure_option,
               tuner='xgb',
               n_trial=1000,
               early_stopping=None,
               log_filename='tuning.log',
               log_db=None):
    import tvm.autotvm as autotvm
    import os
    # create tmp log file
    if log_db is None:
        log_db = log_filename + ".db"

    for i, tsk in enumerate(reversed(tasks)):
        prefix = "[Task %2d/%2d] " % (i+1, len(tasks))

        # create tuner
        if tuner == 'xgb' or tuner == 'xgb-rank':
            tuner_obj = autotvm.tuner.XGBTuner(tsk, loss_type='rank')
        elif tuner == 'ga':
            tuner_obj = autotvm.tuner.GATuner(tsk, pop_size=100)
        elif tuner == 'random':
            tuner_obj = autotvm.tuner.RandomTuner(tsk)
        elif tuner == 'gridsearch':
            tuner_obj = autotvm.tuner.GridSearchTuner(tsk)
        else:
            raise ValueError("Invalid tuner: " + tuner)

        try:
            if os.path.isfile(log_filename):
                tuner_obj.load_history(
                    autotvm.record.load_from_file(log_filename))
        except:
            pass

        # do tuning
        tsk_trial = min(n_trial, len(tsk.config_space))
        import tvm.tvmt
        tuner_obj.tune(n_trial=tsk_trial,
                       early_stopping=early_stopping,
                       measure_option=measure_option,
                       callbacks=[
                           autotvm.callback.progress_bar(
                               tsk_trial, prefix=prefix),
                           autotvm.callback.log_to_file(log_filename),
                           tvm.tvmt.log_to_sqlite(log_db),
                       ])

    # pick best records to a cache file
    # autotvm.record.pick_best(log_filename, log_filename + "best")


def main():
    import test_single_kernel as tsk
    builder, runner = tsk.get_br_parallel()

    batch_size = 1
    num_class = 60
    image_shape = (3, 224, 224)
    data_shape = (batch_size,) + image_shape
    out_shape = (batch_size, num_class)

    mod, params = tvm.relay.testing.resnet.get_workload(
        num_layers=18, batch_size=batch_size, image_shape=image_shape)

    opt_level = 3
    target = tvm.target.cuda()

    tasks = tvm.autotvm.task.extract_from_program(
        mod, params=params, target=target, ops=[tvm.relay.op.get("nn.conv2d"), ])
    measure_option = tvm.autotvm.measure_option(
        builder=builder,
        runner=runner
    )

    tune_tasks(tasks, measure_option, log_filename="test_relay_model.log")

if __name__ == '__main__':
    main()
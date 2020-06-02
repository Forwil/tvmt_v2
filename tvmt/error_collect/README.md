示例依赖修改过的 tvm 代码，修改涉及 c++ 代码，因此需要重新编译，必要时需要重新安装。

依赖 tvm 分支为 [tvmd/eerrrec](https://github.com/huochaitiantang/tvmd/tree/errec)

## 接口

* `tvm.tvmt.log_to_sqlite(filename)`

    获取将一轮编译、运行的结果保存至 `filename` 的 callback function，需要传递给 `tuner.tune(...)` 的 `callback` 参数。

## 常用 SQL

* `select err_no, count(*) as count from logs group by err_no;`

    输出各错误号的数量分布，如

    | err_no | count |
    |--------|-------|
    | 0	| 11213	|
    | 1	| 817	|
    | 4	| 431	|
    | 6	| 7     |

* `select * from logs where err_no=0 order by cost limit 10;`

    输出前10快的各项信息。注意必须用 `err_no=0` 进行约束，因为错误 case 下的 `cost` 为0。

* `select logs.id, err_no, err_text, kvs.k, kvs.v from logs join kvs where logs.err_no!=0 and logs.id=kvs.log_id and logs.workload=kvs.log_workload;`

    输出错误码不为0的错误信息，以及相关的通过 `tvmt.report_json` 保存的所有信息。

## 常见坑

* 程序卡住了，很长时间都没有任何输出

    程序真的挂在某处的可能性应该非常低。比较可能是此时 tuner 正在更新 CostModel，它可能发生在正常训练过程中，也可能发生在加载训练 log 之后，总之可以考虑调高 logger 的级别到 debug，再等上一段时间。

    如果还是极长时间没有反应，连续按几次 ctrl c 结束程序，然后将进程临死前的输出复制下来进行分析。

* `load_history` 也能卡住的吗？

    不能。如果 log 行数大于 500，会在 `load_history` 过程中根据 log 调整 model，参考 `XGBoostCostModel.fit_log`。

## tvm 内部改动

* 增加并使用 `tvmt.report_json` packed_function

    通过 `tvm._ffi.register_func("tvmt.report_json")` 注册一个接受 json 字符串或者 python 对象的函数，这个函数可以在 c++ 和 python 端被调用，在 tvm 中多出模糊化报错信息的地方，都可以加上这个函数。函数将报错信息保存到全局变量中（保证该全局变量中的信息全部都指向同一任务实例）。

* 增加 sqlite log 方式，保存每一个任务实例。设计的数据库为
    * logs

        保存和一般 log 中基本相同的信息，提出部分字段为表头（如 target、cost），以方便查询。

        | *id* | task_name | target | *workload* | config | err_no | trial_no | cost | err_text |
        |------|-----------|--------|------------|--------|--------|----------|------|----------|
        | 与 Config 实体在原一维空间的位置对应 | `tvm.autotvm.task.create` 的第一个参数 | 比如 cuda，和一般 log 中一样 | 序列化的 task 信息，主要包含 task 对应 function 的参数、Tensor shape、dtype 等信息。和一般 log 中一样 | 包含了一个 config 的完整信息，比如各个 knob 的取值，和一般 log 中一样 | 错误编号，参考 #1 中的总结 | 该 workload 下 tune 的第几次尝试 | 去掉最快、最慢后的平均 cost | 报错信息，和一般 log 中一样（不一定完整） |

    * kvs

        保存通过 `tvmt.report_json` packed_function 所保存的额外信息，保存形式为 k-v pair。

        | *id* | log_id | log_workload | k | v |
        |------|--------|--------------|---|---|
        | 编号，不重要 | foreign key -> `logs.id` | foreign key -> `logs.workload` | 字段名 | 字段值 |

    * best_logs (view)

        获得各个不同 `workload` 中速度最快的记录信息。

* executor 修改

    * 多进程

        在 tvmd 的 tvm.tvmt 中，用 monkey patch 修改了原 executor 相关接口，以提供额外错误信息从 worker 进程返回到主进程的能力。

        tvm 尝试并行处理多个 kernel 实例，如并行编译、并行运行，然后再汇总结果。并行时原理为多进程，采用进程间通信。所以并不能简单地用全局变量来各子进程的信息。

        目前的 tvm 中的 Builder、Runner 内部都使用 Executor 来管理多进程任务的派发和返回结果的聚集。对于每一个 batch 的 input，都为其建立一个用来做进程间通信的 queue（python 的多进程通信机制），在子进程中按照超时时间往 queue 中放入 Timeout，如果子进程运行正常，则会在 Timeout 之前将正常返回结果放入 queue。

        父进程启动一个 batch 的并行任务后，会在每个任务对应的 future 对象上调用 get 方法，其中会根据从 queue 取出的第一个对象类型情况获知是否 Timeout，如果运行正常，就能正常拿到结果。

        我们的修改是，替换 `tvm.autotvm.measure.local_executor._execute_func` 和 `tvm.autotvm.measure.local_executor.LocalFuture.get`，前者额外在放入目标对象前先放入当前子进程下用 `tvmt.report_json` 保存的信息，后者则先取出这个对象并放入父进程的全局变量中，再执行剩下的正常操作。

        最终在 callback 时读取全局变量即可拿到每个子进程所返回的信息。

    * 单进程

        单进程仅仅针对 build、runner。

        在实际 build、run 的过程中，增加部分都是向同一个全局变量 `current_tvmt_msg` 添加信息，即需要保证在进入 build 或者 run 的时候，该全局变量为空，并且在 build 或 run 之后，全局变量被使用（即写进“主进程”的 `current_batch_tvmt_msg` 全局变量）。单进程 executor 的做法是，在 `submit` 种判断如果为单进程模式，则直接运行任务，并使用 `class LocalFutureNoFork` 将结果返回。

        因此这里 monkey patch 在执行 `submit` 前判断是否为单进程模式，如果是，则清理 `current_tvmt_msg`，并在原 `submit` 运行结束后把 `current_tvmt_msg` 放入 `current_batch_tvmt_msg`。
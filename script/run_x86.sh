export TVM_NUM_THREADS=1
for i in `cat list`
do
    python run_relay.py x86/$i -d x86 >> x86_thread1.log
done


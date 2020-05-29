for i in `cat list`
do
    python run_relay.py gpu/$i -d gpu >> gpu_t4.log
done


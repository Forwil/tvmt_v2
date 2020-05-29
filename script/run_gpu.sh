for i in `cat list`
do
    python run_relay_speed.py gpu/$i -d gpu >> gpu_t4.log
done


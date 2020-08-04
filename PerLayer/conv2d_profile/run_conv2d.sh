
export TVM_NUM_THREADS=1

#python3 conv2d_profile.py -d x86 -t 200 -b 1 -f x86 -p true
#python3 conv2d_profile.py -d x86 -t 10 -b 1 -f x86 -p true
python3 conv2d_profile.py -d x86-avx2 -t 200 -b 1 -f x86-avx2 -p true
#python3 conv2d_profile.py -d x86-avx512 -t 1 -b 1 -f x86-avx512 -p true


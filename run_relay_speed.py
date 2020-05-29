from utils import *

if __name__ == "__main__":
    import argparse
    import os
    parser = argparse.ArgumentParser(description = "run relay model")
    parser.add_argument("relay", help = "relay model path")
    parser.add_argument("-d", "--device", default="x86", choices=["gpu","x86"])
    arg = parser.parse_args() 
    graph, lib, params = get_model(arg.relay)
    ctx = create_ctx(arg.device) 
    time = speed(graph, lib, params, ctx)
    name = os.path.basename(arg.relay)
    print("%s, %.2f" % (name, time))

from utils import *

if __name__ == "__main__":
    import argparse
    import os
    parser = argparse.ArgumentParser(description = "run relay model")
    parser.add_argument("relay", help = "relay model path")
    parser.add_argument("-d", "--device", default="x86", choices=["gpu","x86"])
    parser.add_argument("-p", "--profile", default="false", type = str)
    arg = parser.parse_args() 
    graph, lib, params = get_model(arg.relay)
    ctx = create_ctx(arg.device) 
    #
    if arg.profile == 'false':
        time = speed(graph, lib, params, ctx)
    elif arg.profile == 'true':
        time = speed_profile(graph, lib, params, ctx)
    #
    name = os.path.basename(arg.relay)
    print("%s, %.2f" % (name, time))

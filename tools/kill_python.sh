ps -ef | grep yuf | grep rpc_server | grep -v grep | awk '{print $2}' | xargs kill -9


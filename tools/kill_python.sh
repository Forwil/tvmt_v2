ps -ef | grep yuf | grep tune | grep -v grep | awk '{print $2}' | xargs kill -9


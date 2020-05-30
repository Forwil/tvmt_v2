ps -ef | grep yuf | grep python | grep -v grep | awk '{print $2}' | xargs kill -9


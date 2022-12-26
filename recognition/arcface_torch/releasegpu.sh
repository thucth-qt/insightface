ps -ef | grep train | grep -v grep | awk '{print kill -9 }' | sh

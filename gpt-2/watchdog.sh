#!/bin/bash
#set -x

timeout="${1:-600}"
cmd="${2:-echo killed}"
truncate_heartbeat_file_to_n_lines=100000

#heartbeat_file="${2:-pong.txt}"
heartbeat_file="pong.txt"

touch "$heartbeat_file"

while true
do
  sleep 1
  for pid in `bash watchpids.sh`
  do
    if kill -0 $pid > /dev/null 2>&1
    then
      latest=`bash watchnonce.sh $pid`
      if [ -z $latest ]
      then
        latest=`etc/utcnow`
      fi
      elapsed=`etc/utcnow $latest`
      echo $pid $elapsed 
      if [ $elapsed -gt $timeout ]
      then
        echo "$(date) kill -9 $pid"
        kill -9 $pid > /dev/null 2>&1
        tail -n $truncate_heartbeat_file_to_n_lines "$heartbeat_file" | sponge "$heartbeat_file"
        set -x
        $cmd
        set +x
      fi
    fi
  done
done

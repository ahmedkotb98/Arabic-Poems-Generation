cat pong.txt | egrep 'pid[0-9]+' -o | egrep '[0-9]+' -o | sort | uniq

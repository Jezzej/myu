#!/bin/bash

echo "Starting Mininet with custom topology..."

# Launch Mininet with your topology, connect to controller at 127.0.0.1:6633
sudo mn --custom ./topology.py --topo demotopo --controller=remote,ip=127.0.0.1,port=6633 --mac --link=tc

# Wait for Mininet to start
sleep 3

# Send traffic in background (ping, iperf)
echo "Sending test traffic..."

xterm -e "mnexec h1 ping -c 10 h3" &
xterm -e "mnexec h2 iperf -c h4 -t 15" &
xterm -e "mnexec h5 iperf -s" &

echo "Traffic tests started. Mininet CLI is available for more testing."


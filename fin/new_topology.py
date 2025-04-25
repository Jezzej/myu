from mininet.net import Mininet
from mininet.topolib import TreeTopo
from mininet.node import RemoteController
from mininet.log import setLogLevel
import time
import os

def test_traffic():
    setLogLevel('info')
    
    # 1. Create a topology: Tree with depth=2, fanout=2
    topo = TreeTopo(depth=2, fanout=2)
    net = Mininet(topo=topo, controller=None)

    # 2. Add Ryu as a remote controller (default Ryu port is 6653)
    controller = RemoteController('ryu', ip='127.0.0.1', port=6653)
    net.addController(controller)

    net.start()

    # 3. Get two hosts to simulate traffic
    h1 = net.get('h1')
    h2 = net.get('h2')

    print("[*] Starting iperf server on h1...")
    h1.cmd('iperf -s &')  # Start iperf server on h1

    print("[*] Generating traffic from h2 -> h1 (60 seconds)...")
    h2.cmd('iperf -c 10.0.0.1 -t 60 -i 5 &')  # Start traffic

    # 4. Monitor port stats for 1 minute
    switches = net.switches

    print("[*] Monitoring switch port stats...")
    for i in range(12):
        for sw in switches:
            sw_name = sw.name
            stats = sw.cmd(f"ovs-ofctl dump-ports {sw_name}")
            print(f"\n[Stats for {sw_name}]")
            print(stats)
        time.sleep(5)

    print("[*] Stopping network...")
    net.stop()

if __name__ == '__main__':
    test_traffic()


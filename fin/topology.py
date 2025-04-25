from mininet.topo import Topo

class DemoTopo(Topo):
    def __init__(self):
        Topo.__init__(self)
        
        # Core switch
        s1 = self.addSwitch('s1')
        
        # Aggregate switches
        s2 = self.addSwitch('s2')
        s3 = self.addSwitch('s3')
        
        # Hosts
        host1 = self.addHost('h1', ip='10.0.0.1')
        host2 = self.addHost('h2', ip='10.0.0.2')
        host3 = self.addHost('h3', ip='10.0.0.3')
        host4 = self.addHost('h4', ip='10.0.0.4')
        host5 = self.addHost('h5', ip='10.0.0.5')
        host6 = self.addHost('h6', ip='10.0.0.6')
        
        # Links
        self.addLink(s1, s2)
        self.addLink(s1, s3)
        self.addLink(s2, host1)
        self.addLink(s2, host2)
        self.addLink(s3, host3)
        self.addLink(s3, host4)
        self.addLink(s3, host5)
        self.addLink(s3, host6)

topos = {'demotopo': (lambda: DemoTopo())}


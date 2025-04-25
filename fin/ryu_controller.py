from ryu.base import app_manager
from ryu.controller import ofp_event
from ryu.controller.handler import CONFIG_DISPATCHER, MAIN_DISPATCHER, set_ev_cls
from ryu.ofproto import ofproto_v1_3
from ryu.lib.packet import packet, ethernet, ether_types, ipv4
from ryu.topology import event
from ryu.topology.api import get_switch, get_link
from ryu.lib import hub
import networkx as nx
import joblib
import numpy as np
import logging
import time

# Configure logging
logging.basicConfig(level=logging.INFO)

# Define basic traffic classes (instead of 5G-specific ones)
TRAFFIC_CLASS_HIGH = 0  # High priority traffic
TRAFFIC_CLASS_MEDIUM = 1  # Medium priority traffic
TRAFFIC_CLASS_LOW = 2  # Low priority traffic

class SimpleMLLoadBalancer(app_manager.RyuApp):

    
    OFP_VERSIONS = [ofproto_v1_3.OFP_VERSION]

    def __init__(self, *args, **kwargs):
        super(SimpleMLLoadBalancer, self).__init__(*args, **kwargs)
        self.logger.info("Starting ML-based Load Balancer")
        
        # Data structures for network state
        self.mac_to_port = {}  # Track MAC address locations
        self.network = nx.DiGraph()  # Network topology graph
        self.datapaths = {}  # Store switch objects
        self.congested_links = {}  # Track congested links with per-flow granularity
        
        # Load ML model (in practice, this would load your trained model)
        try:
            self.model = joblib.load('model/rf_5g.joblib')
            self.scaler = joblib.load('model/scaler.joblib')
            self.logger.info("ML model loaded successfully")
        except Exception as e:
            self.logger.error(f"Error loading ML model: {e}")
            
        
        # Statistics tracking
        self.switch_stats = {}  # Track switch statistics
        self.link_latency = {}  # Track latency between switches
        self.traffic_classes = {}  # Track traffic class per flow
        self.flow_stats = {}  # Track per-flow statistics
        
        # Start background stats collection thread
        self.monitor_thread = hub.spawn(self._monitor)
        
        self.logger.info("Initialization complete")

    def _monitor(self):
        """Background thread for periodic statistics collection."""
        while True:
            for dp in self.datapaths.values():
                self.request_port_stats(dp)
            hub.sleep(5)  # Collect stats every 8 seconds

    @set_ev_cls(ofp_event.EventOFPSwitchFeatures, CONFIG_DISPATCHER)
    def switch_features_handler(self, ev):
        """Handle new switch connection."""
        datapath = ev.msg.datapath
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser

        # Store datapath
        self.datapaths[datapath.id] = datapath
        
        # Initialize switch statistics
        self.switch_stats[datapath.id] = {
            'rx_packets': 0,
            'tx_packets': 0,
            'rx_dropped': 0,
            'tx_dropped': 0,
            'tx_bytes': 0,
            'rx_bytes': 0,
            'latency': 5  # Default latency (ms)
        }
        
        # Install table-miss flow entry (forwards unmatched packets to controller)
        match = parser.OFPMatch()
        actions = [parser.OFPActionOutput(ofproto.OFPP_CONTROLLER, ofproto.OFPCML_NO_BUFFER)]
        self.add_flow(datapath, 0, match, actions)
        
        self.logger.info(f"Switch {datapath.id} connected")
        
        # Request initial port statistics
        self.request_port_stats(datapath)

    def add_flow(self, datapath, priority, match, actions, buffer_id=None, timeout=0):
        """Add a flow entry to the switch."""
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser
        
        # Create instruction set with actions
        inst = [parser.OFPInstructionActions(ofproto.OFPIT_APPLY_ACTIONS, actions)]
        
        # Create flow modification message
        if buffer_id:
            mod = parser.OFPFlowMod(datapath=datapath, buffer_id=buffer_id,
                                   priority=priority, match=match, instructions=inst,
                                   hard_timeout=timeout)
        else:
            mod = parser.OFPFlowMod(datapath=datapath, priority=priority,
                                   match=match, instructions=inst,
                                   hard_timeout=timeout)
                                   
        # Send message to switch
        datapath.send_msg(mod)
        self.logger.debug(f"Flow added on switch {datapath.id}, priority={priority}, timeout={timeout}")

    def request_port_stats(self, datapath):
        """Request port statistics from a switch."""
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser
        req = parser.OFPPortStatsRequest(datapath, 0, ofproto.OFPP_ANY)
        datapath.send_msg(req)

    @set_ev_cls(ofp_event.EventOFPPortStatsReply, MAIN_DISPATCHER)
    def port_stats_reply_handler(self, ev):
        """Process port statistics from switches."""
        body = ev.msg.body
        dpid = ev.msg.datapath.id
        
        # Collect stats from all ports
        rx_packets = tx_packets = rx_dropped = tx_dropped = tx_bytes = rx_bytes = 0
        
        for stat in body:
            rx_packets += stat.rx_packets
            tx_packets += stat.tx_packets
            rx_dropped += stat.rx_dropped
            tx_dropped += stat.tx_errors
            tx_bytes += stat.tx_bytes
            rx_bytes += stat.rx_bytes
        
        # Simulate latency calculation
        latency = min(20, (tx_bytes + rx_bytes) / (10**7) if (tx_bytes + rx_bytes) > 0 else 5)
        
        # Update switch statistics
        self.switch_stats[dpid] = {
            'rx_packets': rx_packets,
            'tx_packets': tx_packets,
            'rx_dropped': rx_dropped,
            'tx_dropped': tx_dropped,
            'tx_bytes': tx_bytes,
            'rx_bytes': rx_bytes,
            'latency': latency
        }
        
        # Update congestion status
        self._update_congestion(dpid)
        
        self.logger.info(f"[STATS] Switch {dpid}: RX={rx_packets}, TX={tx_packets}, "
                        f"RX_bytes={rx_bytes}, TX_bytes={tx_bytes}, "
                        f"Drops={rx_dropped+tx_dropped}, Latency={latency:.2f}ms")
    
    def _update_congestion(self, dpid):
        """Update congestion status based on switch metrics."""
        stats = self.switch_stats[dpid]
        
        # Simple congestion detection based on various metrics
        is_congested = False
        
        # Check if the switch is experiencing high traffic
        if stats['tx_bytes'] > 8000000 or stats['rx_bytes'] > 8000000:
            is_congested = True
            self.logger.info(f"[CONGESTION] Switch {dpid} reports high bandwidth usage")
            
        # Check if the switch has high latency
        if stats['latency'] > 10:
            is_congested = True
            self.logger.info(f"[CONGESTION] Switch {dpid} reports high latency")
            
        # Check if the switch has packet loss
        if (stats['rx_dropped'] + stats['tx_dropped']) > 100:
            is_congested = True
            self.logger.info(f"[CONGESTION] Switch {dpid} reports packet loss")
            
        # For all outgoing links from this switch
        for neighbor in self.network.neighbors(dpid):
            link_key = f"{dpid}-{neighbor}"
            
            # Instead of marking the entire link as congested, we keep track of congestion level
            self.congested_links[link_key] = is_congested
            
            if is_congested:
                # Update the congestion level in the network graph
                if (dpid, neighbor) in self.network.edges():
                    self.network[dpid][neighbor]['congestion_level'] = 5  # Scale from 0-10
            else:
                # Reset congestion level
                if (dpid, neighbor) in self.network.edges():
                    self.network[dpid][neighbor]['congestion_level'] = 0
        
    @set_ev_cls(event.EventSwitchEnter)
    def handle_switch_enter(self, ev):
        """Handle new switch connection and discover topology."""
        # Get all switches
        switches = get_switch(self, None)
        for switch in switches:
            self.network.add_node(switch.dp.id)
            self.datapaths[switch.dp.id] = switch.dp
            
        # Get all links
        links = get_link(self, None)
        for link in links:
            self.network.add_edge(link.src.dpid, link.dst.dpid, 
                                 port=link.src.port_no, 
                                 congestion_level=0,
                                 latency=5)  # Initialize with 5ms latency
            
            # Initialize link latency tracking
            link_key = f"{link.src.dpid}-{link.dst.dpid}"
            self.link_latency[link_key] = 5  # ms
            self.congested_links[link_key] = False  # Initialize as not congested
            
        self.logger.info(f"[TOPOLOGY] Network: {len(switches)} switches, {len(links)} links")
        
        # Request stats from all switches
        for dp in self.datapaths.values():
            self.request_port_stats(dp)

    def predict_congestion(self, features, traffic_class, flow_key):
        """
        Use ML model to predict congestion based on network features and traffic class.
        Also consider flow-specific history.
        """
        try:
            # Get flow history to make prediction more specific
            flow_history = self.flow_stats.get(flow_key, {})
            packet_count = flow_history.get('packet_count', 0)
            
            # Scale features (in real implementation)
            scaled_features = self.scaler.transform(features)
            
            # Make prediction (0=normal, 1=congested)
            prediction = self.model.predict(scaled_features)[0]
            
            # For high priority traffic, be more conservative about congestion
            if traffic_class == TRAFFIC_CLASS_HIGH and features[0][0] > 1000:
                prediction = 1  # Force congestion prediction for sensitive traffic
                
            # For flows with high packet count, be more likely to predict congestion
            if packet_count > 1000:
                prediction = 1
                
            if prediction == 0:
                return "NORMAL", False
            else:
                return "CONGESTED", True
                
        except Exception as e:
            self.logger.error(f"ML prediction error: {e}")
            return "NORMAL", False  # Default to not congested

    def find_best_path(self, src_dpid, dst_dpid, traffic_class, flow_key=None):
        """Find optimal path avoiding congested links, considering traffic class requirements."""
        if not nx.has_path(self.network, src_dpid, dst_dpid):
            return None
            
        # Create a copy of the network for path calculation
        graph = self.network.copy()
        
        # Adjust weights based on traffic class
        for u, v, data in graph.edges(data=True):
            link_key = f"{u}-{v}"
            congestion_level = data.get('congestion_level', 0)
            link_latency = self.link_latency.get(link_key, 5)
            
            # Check if this link is congested
            is_link_congested = self.congested_links.get(link_key, False)
            
            if is_link_congested:
                # Heavily penalize congested links
                base_weight = 1000
            else:
                base_weight = 1 + congestion_level
                
            # Adjust weight based on traffic class requirements
            if traffic_class == TRAFFIC_CLASS_HIGH:
                # High priority: prefer low latency paths
                graph[u][v]['weight'] = base_weight * link_latency * 2
            elif traffic_class == TRAFFIC_CLASS_MEDIUM:
                # Medium priority: balanced approach
                graph[u][v]['weight'] = base_weight * (1 + link_latency/10)
            else:  # TRAFFIC_CLASS_LOW
                # Low priority: just use base weight
                graph[u][v]['weight'] = base_weight
                
        try:
            # Find shortest path using adjusted weights
            path = nx.shortest_path(graph, src_dpid, dst_dpid, weight='weight')
            return path
        except nx.NetworkXNoPath:
            return None

    def get_features_for_prediction(self, dpid, flow_key=None):
        """Extract features for ML congestion prediction."""
        # Default values if stats aren't available
        if dpid not in self.switch_stats:
            return np.array([[0, 0, 0, 0, 0, 0, 0]])
            
        stats = self.switch_stats[dpid]
        
        # Get flow-specific stats if available
        flow_stats = self.flow_stats.get(flow_key, {})
        flow_packet_count = flow_stats.get('packet_count', 0)
        
        # Format features in the order expected by the model
        # Now we include some flow-specific information to make predictions more accurate
        features = np.array([[
            stats['rx_packets'],
            stats['tx_packets'],
            stats['tx_dropped'],
            stats['tx_bytes'],
            stats['rx_dropped'],
            flow_packet_count,  # New feature: flow-specific packet count
            stats['rx_bytes']
        ]])
        
        return features

    def identify_traffic_class(self, pkt):
        """Identify traffic class based on packet characteristics."""
        # Check for IP layer
        ip_pkt = pkt.get_protocol(ipv4.ipv4)
        if not ip_pkt:
            return TRAFFIC_CLASS_MEDIUM  # Default to medium priority
            
        # Simple classification based on IP fields
        if ip_pkt.tos >= 40:  # Check Type of Service / DSCP fields
            return TRAFFIC_CLASS_HIGH  # High priority traffic
        elif ip_pkt.tos >= 20:
            return TRAFFIC_CLASS_MEDIUM  # Medium priority traffic
        else:
            return TRAFFIC_CLASS_LOW  # Low priority traffic

    def install_path_flows(self, path, src_mac, dst_mac, traffic_class, flow_key=None, timeout=30):
        """Install flow rules along the selected path, optimized for traffic class."""
        if not path or len(path) < 2:
            return
            
        # Adjust timeout based on traffic class
        if traffic_class == TRAFFIC_CLASS_HIGH:
            # Shorter timeout for high priority to allow more frequent re-evaluation
            timeout = max(5, timeout // 3)
        elif traffic_class == TRAFFIC_CLASS_LOW:
            # Longer timeout for low priority to reduce control overhead
            timeout = min(300, timeout * 2)
            
        # Get traffic class name for logging
        traffic_class_str = ["HIGH", "MEDIUM", "LOW"][traffic_class]
        self.logger.info(f"[PATH] Installing {traffic_class_str} path: {path}")
            
        # Install flow on each switch in the path
        for i in range(len(path) - 1):
            current_switch = path[i]
            next_switch = path[i + 1]
            
            # Skip if we don't have the datapath object
            if current_switch not in self.datapaths:
                continue
                
            current_dp = self.datapaths[current_switch]
            parser = current_dp.ofproto_parser
            
            # Find output port for current switch to reach the next switch
            out_port = None
            for u, v, data in self.network.edges(data=True):
                if u == current_switch and v == next_switch:
                    out_port = data['port']
                    break
            
            if out_port is None:
                continue
                
            # Create match for this flow
            match = parser.OFPMatch(eth_src=src_mac, eth_dst=dst_mac)
            actions = [parser.OFPActionOutput(out_port)]
            
            # Install flow with the specified timeout and higher priority for high priority traffic
            priority = 2
            if traffic_class == TRAFFIC_CLASS_HIGH:
                priority = 3  # Higher priority 
                
            self.add_flow(current_dp, priority, match, actions, timeout=timeout)
            
            # Mark this flow as installed on this path segment
            if flow_key:
                link_key = f"{current_switch}-{next_switch}"
                # Store flow details in a structure that tracks which flows use which links
                self.flow_stats.setdefault(flow_key, {})
                self.flow_stats[flow_key]['path'] = path
                self.flow_stats[flow_key]['traffic_class'] = traffic_class
                self.flow_stats[flow_key].setdefault('links', []).append(link_key)
            
            self.logger.info(f"[FLOW] {traffic_class_str}: {src_mac}->{dst_mac} | Switch {current_switch}->{next_switch} | "
                            f"Out port={out_port} | Timeout={timeout}s")

    def update_congestion_status(self, path, is_congested, traffic_class, flow_key=None):
        """Update congestion status for links in the path."""
        if not path or len(path) < 2:
            return
            
        # Traffic class affects how we update congestion
        congestion_weight = 1
        if traffic_class == TRAFFIC_CLASS_HIGH:
            congestion_weight = 2  # High priority traffic has bigger impact
        elif traffic_class == TRAFFIC_CLASS_LOW:
            congestion_weight = 0.5  # Low priority has less impact per flow
            
        for i in range(len(path) - 1):
            u, v = path[i], path[i + 1]
            link_key = f"{u}-{v}"
            
            # Update link latency based on switch latency and congestion
            src_latency = self.switch_stats.get(u, {}).get('latency', 5)
            if is_congested:
                self.link_latency[link_key] = src_latency * 1.5  # Increased latency when congested
            else:
                self.link_latency[link_key] = src_latency
            
            # Update congestion status - but make it specific to this flow
            if is_congested:
                # Mark this link as congested
                self.congested_links[link_key] = True
                
                # Increase congestion level (max 10)
                if (u, v) in self.network.edges():
                    current_level = self.network[u][v].get('congestion_level', 0)
                    self.network[u][v]['congestion_level'] = min(10, current_level + (2 * congestion_weight))
                    
                    traffic_class_str = ["HIGH", "MEDIUM", "LOW"][traffic_class]
                    self.logger.info(f"[CONGESTION] Link {u}->{v} congested from {traffic_class_str} traffic "
                                    f"(level={self.network[u][v]['congestion_level']:.1f}, "
                                    f"latency={self.link_latency[link_key]:.2f}ms)")
                    
                    # Track which flow contributed to congestion
                    if flow_key:
                        # This helps identify which flows are causing congestion
                        self.logger.info(f"[FLOW-CONGESTION] Flow {flow_key} contributing to congestion on link {link_key}")
            else:
                # Decrease congestion level
                if (u, v) in self.network.edges() and self.network[u][v].get('congestion_level', 0) > 0:
                    decrease = 1 * congestion_weight
                    self.network[u][v]['congestion_level'] = max(0, self.network[u][v]['congestion_level'] - decrease)
                    
                    if self.network[u][v]['congestion_level'] < 3:
                        self.congested_links[link_key] = False
                        self.logger.info(f"[RECOVERY] Link {u}->{v} recovered from congestion "
                                        f"(level={self.network[u][v]['congestion_level']:.1f})")

    @set_ev_cls(ofp_event.EventOFPPacketIn, MAIN_DISPATCHER)
    def packet_in_handler(self, ev):
        """Handle incoming packets and apply traffic-aware load balancing."""
        msg = ev.msg
        datapath = msg.datapath
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser
        in_port = msg.match['in_port']
        
        # Parse packet
        pkt = packet.Packet(msg.data)
        eth = pkt.get_protocols(ethernet.ethernet)[0]
        
        # Ignore LLDP packets
        if eth.ethertype == ether_types.ETH_TYPE_LLDP:
            return
            
        dst_mac = eth.dst
        src_mac = eth.src
        dpid = datapath.id
        
        # Learn MAC address location
        self.mac_to_port.setdefault(dpid, {})
        self.mac_to_port[dpid][src_mac] = in_port
        
        # Identify traffic class
        traffic_class = self.identify_traffic_class(pkt)
        flow_key = f"{src_mac}-{dst_mac}"
        self.traffic_classes[flow_key] = traffic_class
        
        # Initialize or update flow statistics
        self.flow_stats.setdefault(flow_key, {'packet_count': 0, 'byte_count': 0})
        self.flow_stats[flow_key]['packet_count'] += 1
        self.flow_stats[flow_key]['byte_count'] += len(msg.data)
        
        # Get traffic class name for logging
        traffic_class_str = ["HIGH", "MEDIUM", "LOW"][traffic_class]
        
        # Determine output port
        if dst_mac in self.mac_to_port[dpid]:
            out_port = self.mac_to_port[dpid][dst_mac]
        else:
            out_port = ofproto.OFPP_FLOOD
            
        actions = [parser.OFPActionOutput(out_port)]
        
        # Log packet information
        self.logger.debug(f"[PACKET] {src_mac} -> {dst_mac} on switch {dpid}, port {in_port}, "
                         f"class={traffic_class_str}")
        
        # Only apply ML-based load balancing for known unicast destinations
        if dst_mac != 'ff:ff:ff:ff:ff:ff' and out_port != ofproto.OFPP_FLOOD:
            # Find destination switch for this MAC
            dst_dpid = None
            for switch_id, mac_dict in self.mac_to_port.items():
                if dst_mac in mac_dict:
                    dst_dpid = switch_id
                    break
                    
            if dst_dpid and dst_dpid != dpid:
                # Get features for ML prediction
                features = self.get_features_for_prediction(dpid, flow_key)
                
                # Predict congestion, considering traffic class and flow history
                state, is_congested = self.predict_congestion(features, traffic_class, flow_key)
                
                self.logger.info(f"[ML] Flow {src_mac}->{dst_mac} ({traffic_class_str}): "
                               f"Prediction={state} (Congested={is_congested})")
                
                # For high priority traffic or any congested path, find optimal route
                if is_congested or traffic_class == TRAFFIC_CLASS_HIGH:
                    # Find path optimized for this traffic class
                    path = self.find_best_path(dpid, dst_dpid, traffic_class, flow_key)
                    
                    if path:
                        # Update congestion status - but make it specific to this flow
                        self.update_congestion_status(path, is_congested, traffic_class, flow_key)
                        
                        # Set timeout based on traffic class
                        if traffic_class == TRAFFIC_CLASS_HIGH:
                            timeout = 10  # Short timeout for high priority
                        elif traffic_class == TRAFFIC_CLASS_LOW:
                            timeout = 300  # Long timeout for low priority
                        else:
                            timeout = 30 if is_congested else 120  # Normal timeout for medium
                        
                        # Install flows along the path
                        self.install_path_flows(path, src_mac, dst_mac, traffic_class, flow_key, timeout=timeout)
                        
                        # Update actions to use the first hop of the new path
                        if len(path) > 1:
                            next_switch = path[1]
                            # Find output port to next switch
                            for u, v, data in self.network.edges(data=True):
                                if u == dpid and v == next_switch:
                                    out_port = data['port']
                                    actions = [parser.OFPActionOutput(out_port)]
                                    break
                                    
                        self.logger.info(f"[LOAD BALANCING] {src_mac}->{dst_mac} ({traffic_class_str}) via {path}")
        
        # Install flow entry for this packet
        if out_port != ofproto.OFPP_FLOOD:
            match = parser.OFPMatch(in_port=in_port, eth_dst=dst_mac, eth_src=src_mac)
            
            # Set priority based on traffic class
            priority = 1
            if traffic_class == TRAFFIC_CLASS_HIGH:
                priority = 3  # Highest priority
            elif traffic_class == TRAFFIC_CLASS_MEDIUM:
                priority = 2  # Medium priority
            
            # Set timeout based on traffic type
            if traffic_class == TRAFFIC_CLASS_HIGH:
                timeout = 10  # Short timeout for high priority
            elif traffic_class == TRAFFIC_CLASS_LOW:
                timeout = 300  # Long timeout for low priority
            else:
                # Regular timeout for medium priority, shorter if congested
                timeout = 30 if any(self.congested_links.get(f"{dpid}-{n}", False) for n in self.network.neighbors(dpid)) else 120
            
            if msg.buffer_id != ofproto.OFP_NO_BUFFER:
                self.add_flow(datapath, priority, match, actions, msg.buffer_id, timeout=timeout)
                return
            else:
                self.add_flow(datapath, priority, match, actions, timeout=timeout)
                
        # Send packet out
        data = None
        if msg.buffer_id == ofproto.OFP_NO_BUFFER:
            data = msg.data
            
        out = parser.OFPPacketOut(datapath=datapath, buffer_id=msg.buffer_id,
                                 in_port=in_port, actions=actions, data=data)
        datapath.send_msg(out)






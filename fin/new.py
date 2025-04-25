from ryu.base import app_manager
from ryu.controller import ofp_event
from ryu.controller.handler import CONFIG_DISPATCHER, MAIN_DISPATCHER, set_ev_cls
from ryu.ofproto import ofproto_v1_3
from ryu.lib.packet import packet, ethernet, ether_types
from ryu.topology import event
from ryu.topology.api import get_switch, get_link
import networkx as nx
import joblib
import numpy as np
import time
import psutil

class MLLoadBalancingController(app_manager.RyuApp):
    OFP_VERSIONS = [ofproto_v1_3.OFP_VERSION]

    def __init__(self, *args, **kwargs):
        super(MLLoadBalancingController, self).__init__(*args, **kwargs)
        try:
            self.model = joblib.load('model/rf_5g.joblib')  # Correct path for the model
            self.logger.info("ML model loaded successfully")
            print(f"Model {self.model}")
        except Exception as e:
            self.logger.error(f"Error loading ML model: {e}")
            self.model = None

        try:
            self.scaler = joblib.load('model/scaler.joblib')  # Correct path for the scaler
            self.logger.info("Scaler loaded successfully")
            print(f"Scaler {self.scaler}")
        except Exception as e:
            self.logger.error(f"Error loading scaler: {e}")
            self.scaler = None

        self.mac_to_port = {}
        self.net = nx.DiGraph()
        self.datapaths = {}
        self.switch_stats = {}
        self.link_stats = {}
        self.congested_links = set()
        self.prev_context_switches = psutil.cpu_stats().ctx_switches
        self.prev_time = time.time()

    @set_ev_cls(ofp_event.EventOFPSwitchFeatures, CONFIG_DISPATCHER)
    def switch_features_handler(self, ev):
        datapath = ev.msg.datapath
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser
        dpid = datapath.id

        self.datapaths[dpid] = datapath
        self.switch_stats[dpid] = {
            'OVS_RX_packets': 0,
            'OVS_TX_packets': 0,
            'OVS_RX_dropped': 0,
            'TC_TX_dropped': 0,
            'TC_TX_bytes': 0,
            'TC_RX_bytes': 0,
            'ContextSwitchesPerSecond': 0
        }
        
        self.logger.info(f"Switch {dpid} connected")
        
        match = parser.OFPMatch()
        actions = [parser.OFPActionOutput(ofproto.OFPP_CONTROLLER,
                                          ofproto.OFPCML_NO_BUFFER)]
        self.add_flow(datapath, 0, match, actions)
        self.send_port_stats_request(datapath)

    def add_flow(self, datapath, priority, match, actions, buffer_id=None, timeout=0):
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser
        
        inst = [parser.OFPInstructionActions(ofproto.OFPIT_APPLY_ACTIONS, actions)]
        
        if buffer_id:
            mod = parser.OFPFlowMod(datapath=datapath, buffer_id=buffer_id,
                                    priority=priority, match=match, instructions=inst,
                                    hard_timeout=timeout)
        else:
            mod = parser.OFPFlowMod(datapath=datapath, priority=priority,
                                    match=match, instructions=inst,
                                    hard_timeout=timeout)
        
        datapath.send_msg(mod)

    def send_port_stats_request(self, datapath):
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser
        
        req = parser.OFPPortStatsRequest(datapath, 0, ofproto.OFPP_ANY)
        datapath.send_msg(req)

    @set_ev_cls(ofp_event.EventOFPPortStatsReply, MAIN_DISPATCHER)
    def port_stats_reply_handler(self, ev):
        body = ev.msg.body
        datapath = ev.msg.datapath
        dpid = datapath.id
        
        current_time = time.time()
        
        current_context_switches = psutil.cpu_stats().ctx_switches
        time_diff = current_time - self.prev_time
        
        if time_diff > 0:
            ctx_per_sec = (current_context_switches - self.prev_context_switches) / time_diff
        else:
            ctx_per_sec = 0
            
        self.prev_context_switches = current_context_switches
        self.prev_time = current_time
        
        switch_rx_packets = 0
        switch_tx_packets = 0
        switch_rx_dropped = 0
        tc_tx_dropped = 0
        tc_tx_bytes = 0
        tc_rx_bytes = 0
        
        for stat in body:
            switch_rx_packets += stat.rx_packets
            switch_tx_packets += stat.tx_packets
            switch_rx_dropped += stat.rx_dropped
            tc_tx_dropped += stat.tx_errors
            tc_tx_bytes += stat.tx_bytes
            tc_rx_bytes += stat.rx_bytes
            
        self.switch_stats[dpid] = {
            'OVS_RX_packets': switch_rx_packets,
            'OVS_TX_packets': switch_tx_packets,
            'OVS_RX_dropped': switch_rx_dropped,
            'TC_TX_dropped': tc_tx_dropped,
            'TC_TX_bytes': tc_tx_bytes,
            'TC_RX_bytes': tc_rx_bytes,
            'ContextSwitchesPerSecond': ctx_per_sec
        }

       # Output the updated stats for the switch 
       # This will help you see how stats are changing over time.
       self.logger.info(f"Switch {dpid} Stats: {self.switch_stats[dpid]}")

       # Send periodic requests to keep stats updated 
       datapath.send_msg(datapath.ofproto_parser.OFPPortStatsRequest(datapath, 0, datapath.ofproto.OFPP_ANY))

    @set_ev_cls(event.EventSwitchEnter)
    def handle_switch_enter(self, ev):
       switches = get_switch(self, None)
       self.logger.info("Discovered Switches:")
       
       for switch in switches:
           self.logger.info(f"Switch DPID: {switch.dp.id}")
           self.net.add_node(switch.dp.id)
           self.datapaths[switch.dp.id] = switch.dp

       links = get_link(self, None)
       self.logger.info("Discovered Links:")
       
       for link in links:
           self.logger.info(f"Link: {link.src.dpid}->{link.dst.dpid} Port: {link.src.port_no}")
           self.net.add_edge(link.src.dpid, link.dst.dpid,
                             port=link.src.port_no,
                             congestion_level=0)
           self.link_stats[f"{link.src.dpid}-{link.dst.dpid}"] = {
               'OVS_RX_packets': 0,
               'OVS_TX_packets': 0,
               'OVS_RX_dropped': 0,
               'TC_TX_dropped': 0,
               'TC_TX_bytes': 0,
               'TC_RX_bytes': 0,
               'ContextSwitchesPerSecond': 0
           }

       # Request stats from all switches 
       for dp in self.datapaths.values():
           self.send_port_stats_request(dp)

    def get_network_features(self, src_dpid, dst_dpid):
       path_key = f"{src_dpid}-{dst_dpid}"
       features = {
           'OVS_RX_packets': 0,
           'OVS_TX_packets': 0,
           'TC_TX_dropped': 0,
           'TC_TX_bytes': 0,
           'OVS_RX_dropped': 0,
           'ContextSwitchesPerSecond': 0,
           'TC_RX_bytes': 0
       }
       
       if path_key in self.link_stats:
           features.update(self.link_stats[path_key])
       elif src_dpid in self.switch_stats:
           features.update(self.switch_stats[src_dpid])
       
       # Log the features being used for prediction 
       self.logger.info(f"Features for prediction from {src_dpid} to {dst_dpid}: {features}")
       
       return np.array([[
           features['OVS_RX_packets'],
           features['OVS_TX_packets'],
           features['TC_TX_dropped'],
           features['TC_TX_bytes'],
           features['OVS_RX_dropped'],
           features['ContextSwitchesPerSecond'],
           features['TC_RX_bytes']
       ]])

    def predict_congestion(self, features):
       try:
           scaled_features = self.scaler.transform(features)
           
           # Log the scaled features before prediction 
           self.logger.info(f"Scaled Features: {scaled_features}")
           
           prediction = self.model.predict(scaled_features)[0]
           
           if prediction == 0:
               return "ACTIVE", False
           else:
               return "DROPPED", True

       except Exception as e:
           # Log any errors during prediction 
           self.logger.error(f"Error in congestion prediction: {e}")
           return "ACTIVE", False

    def find_optimal_path(self, src_dpid, dst_dpid):
       if not nx.has_path(self.net, src_dpid, dst_dpid):
           return None

       graph = self.net.copy()
       
       for u, v in graph.edges():
           link_key = f"{u}-{v}"
           
           if (u,v) in self.congested_links:
               graph[u][v]['weight'] = float('inf')   # Set weight to a high value to avoid congested links.
               
       try:
          path=nx.shortest_path(graph ,src_dpid ,dst_dpid ,weight='weight')
          return path

       except nx.NetworkXNoPath:
          return None

    def install_path_flows(self,path ,src_mac ,dst_mac ,datapath ,in_port ,out_port ,timeout=30):
      if not path or len(path) <2 :
          return
      
      ofproto=datapath.ofproto 
      parser=datapath.ofproto_parser
      
      for i in range(len(path)-1):
          current_switch=path[i]
          next_switch=path[i+1]
          
          if current_switch not inself.datapaths:
              continue
            
          current_dp=self.datapaths[current_switch]
          out_port=None 
          
          for u,v,data inself.net.edges(data=True):
              if u==current_switch and v==next_switch:
                  out_port=data['port']
                  break
            
          if out_port is None :
              continue
            
          match=parser.OFPMatch(eth_src=src_mac ,eth_dst=dst_mac)
          actions=[parser.OFPActionOutput(out_port)]
          
          # Install flow on this switch 
         self.add_flow(current_dp ,2 ,match ,actions ,timeout)

    def update_link_stats(self,path,is_congested):
      if not path or len(path) <2 :
          return
      
      for i in range(len(path)-1):
          u,v=path[i],path[i+1]
          link_key=f"{u}-{v}"
          
          if is_congested:
              # Mark the link as congested 
             self.congested_links.add((u,v))
              
              if (u,v)inself.net.edges():
                 self.net[u][v]['congestion_level']=min(10,self.net[u][v].get('congestion_level',0)+2)
          
          else:
              if (u,v)inself.congested_links:
                 self.congested_links.remove((u,v))
              
              if (u,v)inself.net.edges() andself.net[u][v].get('congestion_level',0)>0:
                 self.net[u][v]['congestion_level']-=1
            
          if u inself.switch_stats and link_key inself.link_stats:
              switch_stats=self.switch_stats[u]
             self.link_stats[link_key]={
                  'OVS_RX_packets':switch_stats['OVS_RX_packets']//2 ,
                  'OVS_TX_packets':switch_stats['OVS_TX_packets']//2 ,
                  'OVS_RX_dropped' :switch_stats['OVS_RX_dropped']//2 ,
                  'TC_TX_dropped' :switch_stats['TC_TX_dropped']//2 ,
                  'TC_TX_bytes' :switch_stats['TC_TX_bytes']//2 ,
                  'TC_RX_bytes' :switch_stats['TC_RX_bytes']//2 ,
                  'ContextSwitchesPerSecond' :switch_stats['ContextSwitchesPerSecond']
              }
              
              if is_congested:
                 self.link_stats[link_key]['TC_TX_dropped']*=1.5 
                 self.link_stats[link_key]['OVS_RX_dropped']*=1.5 

    @set_ev_cls(ofp_event.EventOFPPacketIn ,MAIN_DISPATCHER)
    def packet_in_handler(self ,ev):
      msg=ev.msg 
      datapath=msg.datapath 
      ofproto=datapath.ofproto 
      parser=datapath.ofproto_parser 
      in_port=msg.match['in_port']
      
      pkt=packet.Packet(msg.data) 
      eth=pkt.get_protocols(ethernet.ethernet)[0]
      
      if eth.ethertype==ether_types.ETH_TYPE_LLDP :
          return
      
      dst=eth.dst 
      src=eth.src 
      dpid=datapath.id 
      
      # Store MAC addresses and their corresponding ports 
     self.mac_to_port.setdefault(dpid,{})
     self.mac_to_port[dpid][src]=in_port
      
      if dst inself.mac_to_port[dpid]:
          out_port=self.mac_to_port[dpid][dst]
      
      else :
          out_port=ofproto.OFPP_FLOOD
      
      actions=[parser.OFPActionOutput(out_port)]
      
      match=parser.OFPMatch(in_port=in_port ,eth_dst=dst ,eth_src=src)
      
      data=None 
      
      if msg.buffer_id==ofproto.OFP_NO_BUFFER :
          data=msg.data
      
      out=parser.OFPPacketOut(datapath=datapath ,buffer_id=msg.buffer_id ,
                              in_port=in_port ,actions=actions ,data=data)
      
      datapath.send_msg(out)

     # Check ML model and scaler before processing packets.
     ifself.model andself.scaler and len(self.net.edges()) > 0 :
         src_dpid=None 
         dst_dpid=None 

         for node inself.net.nodes():
             if node inself.mac_to_port and src inself.mac_to_port[node]:
                 src_dpid=node
            
             if node inself.mac_to_port and dst inself.mac_to_port[node]:
                 dst_dpid=node

         if src_dpid and dst_dpid :
             features=self.get_network_features(src_dpid,dst_dpid)
             state,is_congested=self.predict_congestion(features)

             # Update link statistics based on congestion prediction.
            self.update_link_stats(self.find_optimal_path(src_dpid,dst_dpid),is_congested)

             optimal_path=self.find_optimal_path(src_dpid,dst_dpid)

             if optimal_path :
                 # Install flows along the optimal path.
                self.install_path_flows(optimal_path ,src ,dst ,datapath ,in_port,out_port)
             else :
                 # Log a warning when no path is found.
                self.logger.warning(f"No path found from {src_dpid} to {dst_dpid}")


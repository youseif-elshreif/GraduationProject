# Network Track Documentation - IDS-AI System

## Table of Contents
1. [Overview](#overview)
2. [Network Architecture](#network-architecture)
3. [Traffic Capture Methods](#traffic-capture-methods)
4. [Network Flow Analysis](#network-flow-analysis)
5. [Packet Processing Pipeline](#packet-processing-pipeline)
6. [Network Monitoring Tools](#network-monitoring-tools)
7. [Network Actions & Response](#network-actions--response)
8. [Security Implementation](#security-implementation)
9. [Performance Optimization](#performance-optimization)
10. [Network Protocols](#network-protocols)
11. [Deployment Strategies](#deployment-strategies)
12. [Monitoring & Alerting](#monitoring--alerting)
13. [Troubleshooting](#troubleshooting)
14. [Best Practices](#best-practices)

## Overview

The network component of the IDS-AI system is responsible for capturing, analyzing, and responding to network traffic in real-time. It operates at the network infrastructure level, providing comprehensive visibility into network communications while maintaining minimal impact on network performance.

### Core Functions
- **Traffic Capture**: Passive monitoring of network traffic through SPAN/mirror ports
- **Flow Generation**: Convert raw packets into structured network flow records
- **Feature Extraction**: Generate statistical and behavioral features from network flows
- **Real-time Processing**: Process network data with minimal latency
- **Network Actions**: Execute security responses (IP blocking, port filtering)
- **Protocol Analysis**: Deep inspection of various network protocols

### Key Requirements
- **High Throughput**: Handle multi-gigabit network traffic
- **Low Latency**: Real-time processing with sub-second response times
- **Scalability**: Support for multiple network segments and interfaces
- **Reliability**: 24/7 operation with minimal downtime
- **Security**: Secure operation without compromising network integrity

## Network Architecture

### Physical Network Topology

```
                    Internet
                       │
                 ┌─────┴─────┐
                 │  Firewall │
                 └─────┬─────┘
                       │
            ┌──────────┴──────────┐
            │   Core Router       │
            │   (SPAN Port)       │
            └──────────┬──────────┘
                       │
            ┌──────────┴──────────┐
            │  Distribution       │
            │     Switch          │
            └─────┬─────────┬─────┘
                  │         │
        ┌─────────┴───┐   ┌─┴─────────────┐
        │   Access    │   │   Access      │
        │  Switch 1   │   │  Switch 2     │
        └─────┬───────┘   └───────┬───────┘
              │                   │
        ┌─────┴─────┐       ┌─────┴─────┐
        │   End     │       │   End     │
        │  Devices  │       │  Devices  │
        └───────────┘       └───────────┘

              │
              ▼
    ┌─────────────────────┐
    │   IDS-AI System     │
    │   (SPAN Mirror)     │
    │                     │
    │  ┌───────────────┐  │
    │  │ Packet Capture│  │
    │  └───────┬───────┘  │
    │          │          │
    │  ┌───────▼───────┐  │
    │  │ Flow Generator│  │
    │  └───────┬───────┘  │
    │          │          │
    │  ┌───────▼───────┐  │
    │  │  AI Analysis  │  │
    │  └───────┬───────┘  │
    │          │          │
    │  ┌───────▼───────┐  │
    │  │ Response Unit │  │
    │  └───────────────┘  │
    └─────────────────────┘
```

### Logical Network Layers

```python
# Network Stack Layers
class NetworkStack:
    def __init__(self):
        self.layers = {
            'Physical': 'SPAN/Mirror Port Connection',
            'Data Link': 'Ethernet Frame Processing',
            'Network': 'IP Packet Analysis',
            'Transport': 'TCP/UDP Session Tracking',
            'Session': 'Flow State Management',
            'Presentation': 'Protocol Decoding',
            'Application': 'Deep Packet Inspection'
        }
```

## Traffic Capture Methods

### SPAN Port Configuration

```bash
# Cisco Switch SPAN Configuration
configure terminal
monitor session 1 source interface GigabitEthernet1/0/1 - 24
monitor session 1 destination interface GigabitEthernet1/0/48
monitor session 1 filter ip access-group 100
exit

# Access List for Traffic Filtering
access-list 100 permit ip any any
access-list 100 deny icmp any any echo
```

### Mirror Port Setup

```bash
# Linux Bridge Mirror Configuration
# Create bridge interface
brctl addbr br-monitor
brctl addif br-monitor eth0
brctl addif br-monitor eth1

# Enable port mirroring
tc qdisc add dev eth0 handle 1: root prio
tc filter add dev eth0 parent 1: protocol ip prio 1 u32 match ip src 0.0.0.0/0 action mirred egress mirror dev eth2

# Monitor interface configuration
ip link set eth2 up
ip link set eth2 promisc on
```

### TAP Interface Implementation

```python
# network/tap_interface.py
import socket
import struct
from typing import Optional, Callable
import threading
import time

class TAPInterface:
    def __init__(self, interface: str, callback: Callable):
        self.interface = interface
        self.callback = callback
        self.running = False
        self.socket = None
        
    def start_capture(self):
        """Start packet capture on TAP interface"""
        try:
            # Create raw socket
            self.socket = socket.socket(
                socket.AF_PACKET, 
                socket.SOCK_RAW, 
                socket.ntohs(0x0003)
            )
            self.socket.bind((self.interface, 0))
            self.socket.settimeout(1.0)
            
            self.running = True
            capture_thread = threading.Thread(target=self._capture_loop)
            capture_thread.daemon = True
            capture_thread.start()
            
            print(f"Started packet capture on {self.interface}")
            
        except Exception as e:
            print(f"Failed to start capture: {e}")
            raise
    
    def _capture_loop(self):
        """Main packet capture loop"""
        while self.running:
            try:
                packet, addr = self.socket.recvfrom(65536)
                if packet:
                    self.callback(packet, time.time())
            except socket.timeout:
                continue
            except Exception as e:
                print(f"Capture error: {e}")
                break
    
    def stop_capture(self):
        """Stop packet capture"""
        self.running = False
        if self.socket:
            self.socket.close()
```

### PCap Integration

```python
# network/pcap_capture.py
import pcap
import threading
from typing import Dict, Any
import struct
import time

class PCAPCapture:
    def __init__(self, interface: str, filter_str: str = ""):
        self.interface = interface
        self.filter_str = filter_str
        self.pc = None
        self.running = False
        
    def start_capture(self, callback):
        """Start PCap capture with callback"""
        try:
            self.pc = pcap.pcap(name=self.interface, promisc=True, immediate=True)
            
            if self.filter_str:
                self.pc.setfilter(self.filter_str)
            
            self.running = True
            
            def capture_thread():
                for timestamp, packet in self.pc:
                    if not self.running:
                        break
                    callback(packet, timestamp)
            
            thread = threading.Thread(target=capture_thread)
            thread.daemon = True
            thread.start()
            
            print(f"PCap capture started on {self.interface}")
            
        except Exception as e:
            print(f"PCap capture failed: {e}")
            raise
    
    def stop_capture(self):
        """Stop PCap capture"""
        self.running = False
        if self.pc:
            self.pc.close()
```

## Network Flow Analysis

### Flow Record Structure

```python
# network/flow_record.py
from dataclasses import dataclass
from typing import Dict, Any, Optional
import time
import hashlib

@dataclass
class FlowRecord:
    """Network flow record structure"""
    # Flow Identification
    flow_id: str
    src_ip: str
    dst_ip: str
    src_port: int
    dst_port: int
    protocol: str
    
    # Timing Information
    start_time: float
    end_time: float
    duration: float
    
    # Packet Statistics
    fwd_packets: int = 0
    bwd_packets: int = 0
    total_packets: int = 0
    
    # Byte Statistics
    fwd_bytes: int = 0
    bwd_bytes: int = 0
    total_bytes: int = 0
    
    # Flow Characteristics
    fwd_packet_lengths: list = None
    bwd_packet_lengths: list = None
    inter_arrival_times: list = None
    
    # TCP Flags
    tcp_flags: Dict[str, int] = None
    
    # Feature Vector (80 features)
    features: Dict[str, float] = None
    
    def __post_init__(self):
        if self.fwd_packet_lengths is None:
            self.fwd_packet_lengths = []
        if self.bwd_packet_lengths is None:
            self.bwd_packet_lengths = []
        if self.inter_arrival_times is None:
            self.inter_arrival_times = []
        if self.tcp_flags is None:
            self.tcp_flags = {
                'FIN': 0, 'SYN': 0, 'RST': 0, 'PSH': 0,
                'ACK': 0, 'URG': 0, 'ECE': 0, 'CWR': 0
            }
    
    @classmethod
    def generate_flow_id(cls, src_ip: str, dst_ip: str, src_port: int, 
                        dst_port: int, protocol: str) -> str:
        """Generate unique flow identifier"""
        flow_key = f"{src_ip}:{src_port}-{dst_ip}:{dst_port}-{protocol}"
        return hashlib.md5(flow_key.encode()).hexdigest()[:16]
    
    def calculate_features(self) -> Dict[str, float]:
        """Calculate 80 statistical features from flow data"""
        features = {}
        
        # Duration Features
        features['flow_duration'] = self.duration
        features['flow_bytes_s'] = self.total_bytes / max(self.duration, 0.001)
        features['flow_packets_s'] = self.total_packets / max(self.duration, 0.001)
        
        # Packet Length Features
        all_lengths = self.fwd_packet_lengths + self.bwd_packet_lengths
        if all_lengths:
            features['pkt_len_max'] = max(all_lengths)
            features['pkt_len_min'] = min(all_lengths)
            features['pkt_len_mean'] = sum(all_lengths) / len(all_lengths)
            features['pkt_len_std'] = self._calculate_std(all_lengths)
            features['pkt_len_var'] = features['pkt_len_std'] ** 2
        
        # Forward Packet Features
        if self.fwd_packet_lengths:
            features['fwd_pkt_len_max'] = max(self.fwd_packet_lengths)
            features['fwd_pkt_len_min'] = min(self.fwd_packet_lengths)
            features['fwd_pkt_len_mean'] = sum(self.fwd_packet_lengths) / len(self.fwd_packet_lengths)
            features['fwd_pkt_len_std'] = self._calculate_std(self.fwd_packet_lengths)
        
        # Backward Packet Features
        if self.bwd_packet_lengths:
            features['bwd_pkt_len_max'] = max(self.bwd_packet_lengths)
            features['bwd_pkt_len_min'] = min(self.bwd_packet_lengths)
            features['bwd_pkt_len_mean'] = sum(self.bwd_packet_lengths) / len(self.bwd_packet_lengths)
            features['bwd_pkt_len_std'] = self._calculate_std(self.bwd_packet_lengths)
        
        # Inter-Arrival Time Features
        if self.inter_arrival_times:
            features['flow_iat_mean'] = sum(self.inter_arrival_times) / len(self.inter_arrival_times)
            features['flow_iat_std'] = self._calculate_std(self.inter_arrival_times)
            features['flow_iat_max'] = max(self.inter_arrival_times)
            features['flow_iat_min'] = min(self.inter_arrival_times)
        
        # TCP Flag Features
        for flag, count in self.tcp_flags.items():
            features[f'{flag.lower()}_flag_cnt'] = count
        
        # Flow Statistics
        features['tot_fwd_pkts'] = self.fwd_packets
        features['tot_bwd_pkts'] = self.bwd_packets
        features['totlen_fwd_pkts'] = self.fwd_bytes
        features['totlen_bwd_pkts'] = self.bwd_bytes
        
        # Ratios and Derived Features
        if self.total_packets > 0:
            features['down_up_ratio'] = self.bwd_packets / self.total_packets
            features['pkt_size_avg'] = self.total_bytes / self.total_packets
        
        if self.fwd_packets > 0:
            features['fwd_seg_size_avg'] = self.fwd_bytes / self.fwd_packets
        
        if self.bwd_packets > 0:
            features['bwd_seg_size_avg'] = self.bwd_bytes / self.bwd_packets
        
        self.features = features
        return features
    
    def _calculate_std(self, values: list) -> float:
        """Calculate standard deviation"""
        if len(values) < 2:
            return 0.0
        
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / (len(values) - 1)
        return variance ** 0.5
```

### Flow Generation Engine

```python
# network/flow_generator.py
import time
import threading
from collections import defaultdict
from typing import Dict, Callable, Optional
from network.flow_record import FlowRecord
from network.packet_parser import PacketParser

class FlowGenerator:
    def __init__(self, 
                 flow_timeout: float = 600.0,
                 active_timeout: float = 1800.0,
                 flow_callback: Optional[Callable] = None):
        self.flow_timeout = flow_timeout
        self.active_timeout = active_timeout
        self.flow_callback = flow_callback
        
        # Flow tracking
        self.active_flows: Dict[str, FlowRecord] = {}
        self.flow_lock = threading.RLock()
        
        # Packet parser
        self.parser = PacketParser()
        
        # Cleanup thread
        self.cleanup_thread = threading.Thread(target=self._cleanup_expired_flows, daemon=True)
        self.running = True
        self.cleanup_thread.start()
    
    def process_packet(self, packet_data: bytes, timestamp: float):
        """Process incoming packet and update flows"""
        try:
            packet_info = self.parser.parse_packet(packet_data, timestamp)
            if not packet_info:
                return
            
            flow_id = FlowRecord.generate_flow_id(
                packet_info['src_ip'],
                packet_info['dst_ip'],
                packet_info['src_port'],
                packet_info['dst_port'],
                packet_info['protocol']
            )
            
            with self.flow_lock:
                if flow_id not in self.active_flows:
                    # Create new flow
                    flow = FlowRecord(
                        flow_id=flow_id,
                        src_ip=packet_info['src_ip'],
                        dst_ip=packet_info['dst_ip'],
                        src_port=packet_info['src_port'],
                        dst_port=packet_info['dst_port'],
                        protocol=packet_info['protocol'],
                        start_time=timestamp,
                        end_time=timestamp,
                        duration=0.0
                    )
                    self.active_flows[flow_id] = flow
                else:
                    flow = self.active_flows[flow_id]
                
                # Update flow with packet information
                self._update_flow(flow, packet_info, timestamp)
                
                # Check if flow should be exported
                if self._should_export_flow(flow, timestamp):
                    self._export_flow(flow)
                    del self.active_flows[flow_id]
        
        except Exception as e:
            print(f"Error processing packet: {e}")
    
    def _update_flow(self, flow: FlowRecord, packet_info: Dict, timestamp: float):
        """Update flow record with packet information"""
        # Update timing
        if timestamp < flow.start_time:
            flow.start_time = timestamp
        if timestamp > flow.end_time:
            flow.end_time = timestamp
        flow.duration = flow.end_time - flow.start_time
        
        # Determine packet direction
        is_forward = (packet_info['src_ip'] == flow.src_ip and 
                     packet_info['src_port'] == flow.src_port)
        
        # Update packet counts and lengths
        packet_length = packet_info['length']
        if is_forward:
            flow.fwd_packets += 1
            flow.fwd_bytes += packet_length
            flow.fwd_packet_lengths.append(packet_length)
        else:
            flow.bwd_packets += 1
            flow.bwd_bytes += packet_length
            flow.bwd_packet_lengths.append(packet_length)
        
        flow.total_packets = flow.fwd_packets + flow.bwd_packets
        flow.total_bytes = flow.fwd_bytes + flow.bwd_bytes
        
        # Update inter-arrival times
        if len(flow.inter_arrival_times) > 0:
            last_time = flow.inter_arrival_times[-1] if flow.inter_arrival_times else flow.start_time
            iat = timestamp - last_time
            flow.inter_arrival_times.append(iat)
        
        # Update TCP flags
        if 'tcp_flags' in packet_info:
            for flag, value in packet_info['tcp_flags'].items():
                if value:
                    flow.tcp_flags[flag] += 1
    
    def _should_export_flow(self, flow: FlowRecord, current_time: float) -> bool:
        """Determine if flow should be exported"""
        # Flow timeout (no activity)
        if current_time - flow.end_time > self.flow_timeout:
            return True
        
        # Active timeout (total duration)
        if flow.duration > self.active_timeout:
            return True
        
        # TCP connection closed (FIN flags)
        if flow.protocol == 'TCP' and flow.tcp_flags.get('FIN', 0) >= 2:
            return True
        
        return False
    
    def _export_flow(self, flow: FlowRecord):
        """Export completed flow"""
        # Calculate features
        flow.calculate_features()
        
        # Send to callback if provided
        if self.flow_callback:
            try:
                self.flow_callback(flow)
            except Exception as e:
                print(f"Error in flow callback: {e}")
    
    def _cleanup_expired_flows(self):
        """Background thread to clean up expired flows"""
        while self.running:
            current_time = time.time()
            expired_flows = []
            
            with self.flow_lock:
                for flow_id, flow in list(self.active_flows.items()):
                    if self._should_export_flow(flow, current_time):
                        expired_flows.append(flow_id)
                
                for flow_id in expired_flows:
                    flow = self.active_flows.pop(flow_id, None)
                    if flow:
                        self._export_flow(flow)
            
            time.sleep(30)  # Check every 30 seconds
    
    def get_flow_stats(self) -> Dict[str, int]:
        """Get current flow statistics"""
        with self.flow_lock:
            return {
                'active_flows': len(self.active_flows),
                'total_flows': len(self.active_flows)  # Would track historical count
            }
```

## Packet Processing Pipeline

### Packet Parser Implementation

```python
# network/packet_parser.py
import struct
import socket
from typing import Dict, Optional, Any

class PacketParser:
    def __init__(self):
        self.protocols = {
            1: 'ICMP',
            6: 'TCP',
            17: 'UDP'
        }
    
    def parse_packet(self, packet_data: bytes, timestamp: float) -> Optional[Dict[str, Any]]:
        """Parse network packet and extract relevant information"""
        try:
            # Parse Ethernet header
            eth_header = self._parse_ethernet(packet_data)
            if eth_header['ethertype'] != 0x0800:  # IPv4
                return None
            
            # Parse IP header
            ip_offset = 14  # Ethernet header size
            ip_header = self._parse_ip(packet_data[ip_offset:])
            if not ip_header:
                return None
            
            # Parse transport layer
            transport_offset = ip_offset + ip_header['header_length']
            transport_info = {}
            
            if ip_header['protocol'] == 6:  # TCP
                transport_info = self._parse_tcp(packet_data[transport_offset:])
            elif ip_header['protocol'] == 17:  # UDP
                transport_info = self._parse_udp(packet_data[transport_offset:])
            elif ip_header['protocol'] == 1:  # ICMP
                transport_info = self._parse_icmp(packet_data[transport_offset:])
            
            # Combine all information
            packet_info = {
                'timestamp': timestamp,
                'length': len(packet_data),
                'src_ip': ip_header['src_ip'],
                'dst_ip': ip_header['dst_ip'],
                'protocol': self.protocols.get(ip_header['protocol'], 'OTHER'),
                'src_port': transport_info.get('src_port', 0),
                'dst_port': transport_info.get('dst_port', 0),
                'tcp_flags': transport_info.get('flags', {}),
                'ip_header': ip_header,
                'transport_header': transport_info
            }
            
            return packet_info
            
        except Exception as e:
            print(f"Packet parsing error: {e}")
            return None
    
    def _parse_ethernet(self, packet: bytes) -> Dict[str, Any]:
        """Parse Ethernet header"""
        if len(packet) < 14:
            raise ValueError("Packet too short for Ethernet header")
        
        # Unpack Ethernet header
        eth_header = struct.unpack('!6s6sH', packet[:14])
        
        return {
            'dst_mac': ':'.join(f'{b:02x}' for b in eth_header[0]),
            'src_mac': ':'.join(f'{b:02x}' for b in eth_header[1]),
            'ethertype': eth_header[2]
        }
    
    def _parse_ip(self, packet: bytes) -> Dict[str, Any]:
        """Parse IP header"""
        if len(packet) < 20:
            raise ValueError("Packet too short for IP header")
        
        # Unpack IP header
        ip_header = struct.unpack('!BBHHHBBH4s4s', packet[:20])
        
        version = (ip_header[0] >> 4) & 0xF
        if version != 4:
            return None  # Only IPv4 supported
        
        header_length = (ip_header[0] & 0xF) * 4
        
        return {
            'version': version,
            'header_length': header_length,
            'type_of_service': ip_header[1],
            'total_length': ip_header[2],
            'identification': ip_header[3],
            'flags': ip_header[4] >> 13,
            'fragment_offset': ip_header[4] & 0x1FFF,
            'ttl': ip_header[5],
            'protocol': ip_header[6],
            'checksum': ip_header[7],
            'src_ip': socket.inet_ntoa(ip_header[8]),
            'dst_ip': socket.inet_ntoa(ip_header[9])
        }
    
    def _parse_tcp(self, packet: bytes) -> Dict[str, Any]:
        """Parse TCP header"""
        if len(packet) < 20:
            raise ValueError("Packet too short for TCP header")
        
        # Unpack TCP header
        tcp_header = struct.unpack('!HHLLBBHHH', packet[:20])
        
        flags_byte = tcp_header[5]
        flags = {
            'FIN': bool(flags_byte & 0x01),
            'SYN': bool(flags_byte & 0x02),
            'RST': bool(flags_byte & 0x04),
            'PSH': bool(flags_byte & 0x08),
            'ACK': bool(flags_byte & 0x10),
            'URG': bool(flags_byte & 0x20),
            'ECE': bool(flags_byte & 0x40),
            'CWR': bool(flags_byte & 0x80)
        }
        
        return {
            'src_port': tcp_header[0],
            'dst_port': tcp_header[1],
            'sequence': tcp_header[2],
            'acknowledgment': tcp_header[3],
            'header_length': (tcp_header[4] >> 4) * 4,
            'flags': flags,
            'window': tcp_header[6],
            'checksum': tcp_header[7],
            'urgent_pointer': tcp_header[8]
        }
    
    def _parse_udp(self, packet: bytes) -> Dict[str, Any]:
        """Parse UDP header"""
        if len(packet) < 8:
            raise ValueError("Packet too short for UDP header")
        
        # Unpack UDP header
        udp_header = struct.unpack('!HHHH', packet[:8])
        
        return {
            'src_port': udp_header[0],
            'dst_port': udp_header[1],
            'length': udp_header[2],
            'checksum': udp_header[3]
        }
    
    def _parse_icmp(self, packet: bytes) -> Dict[str, Any]:
        """Parse ICMP header"""
        if len(packet) < 8:
            raise ValueError("Packet too short for ICMP header")
        
        # Unpack ICMP header
        icmp_header = struct.unpack('!BBHHH', packet[:8])
        
        return {
            'type': icmp_header[0],
            'code': icmp_header[1],
            'checksum': icmp_header[2],
            'identifier': icmp_header[3],
            'sequence': icmp_header[4]
        }
```

### High-Performance Processing

```python
# network/high_performance_processor.py
import asyncio
import queue
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Optional
import multiprocessing as mp

class HighPerformanceProcessor:
    def __init__(self, 
                 num_workers: int = None,
                 queue_size: int = 10000,
                 batch_size: int = 100):
        self.num_workers = num_workers or mp.cpu_count()
        self.queue_size = queue_size
        self.batch_size = batch_size
        
        # Processing queues
        self.packet_queue = queue.Queue(maxsize=queue_size)
        self.flow_queue = queue.Queue(maxsize=queue_size)
        
        # Worker threads
        self.workers = []
        self.running = False
        
        # Statistics
        self.stats = {
            'packets_processed': 0,
            'flows_generated': 0,
            'packets_dropped': 0,
            'processing_errors': 0
        }
    
    def start(self, packet_processor: Callable, flow_processor: Callable):
        """Start high-performance processing"""
        self.running = True
        
        # Start packet processing workers
        for i in range(self.num_workers):
            worker = threading.Thread(
                target=self._packet_worker,
                args=(packet_processor,),
                name=f"PacketWorker-{i}"
            )
            worker.daemon = True
            worker.start()
            self.workers.append(worker)
        
        # Start flow processing worker
        flow_worker = threading.Thread(
            target=self._flow_worker,
            args=(flow_processor,),
            name="FlowWorker"
        )
        flow_worker.daemon = True
        flow_worker.start()
        self.workers.append(flow_worker)
        
        print(f"Started {self.num_workers + 1} processing workers")
    
    def submit_packet(self, packet_data: bytes, timestamp: float) -> bool:
        """Submit packet for processing"""
        try:
            self.packet_queue.put_nowait((packet_data, timestamp))
            return True
        except queue.Full:
            self.stats['packets_dropped'] += 1
            return False
    
    def submit_flow(self, flow_record) -> bool:
        """Submit flow for processing"""
        try:
            self.flow_queue.put_nowait(flow_record)
            return True
        except queue.Full:
            return False
    
    def _packet_worker(self, processor: Callable):
        """Packet processing worker thread"""
        batch = []
        
        while self.running:
            try:
                # Collect batch of packets
                while len(batch) < self.batch_size and self.running:
                    try:
                        item = self.packet_queue.get(timeout=0.1)
                        batch.append(item)
                    except queue.Empty:
                        break
                
                if batch:
                    # Process batch
                    for packet_data, timestamp in batch:
                        try:
                            processor(packet_data, timestamp)
                            self.stats['packets_processed'] += 1
                        except Exception as e:
                            self.stats['processing_errors'] += 1
                            print(f"Packet processing error: {e}")
                    
                    batch.clear()
                    
            except Exception as e:
                print(f"Worker error: {e}")
    
    def _flow_worker(self, processor: Callable):
        """Flow processing worker thread"""
        while self.running:
            try:
                flow_record = self.flow_queue.get(timeout=1.0)
                processor(flow_record)
                self.stats['flows_generated'] += 1
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Flow processing error: {e}")
    
    def get_stats(self) -> dict:
        """Get processing statistics"""
        return self.stats.copy()
    
    def stop(self):
        """Stop processing"""
        self.running = False
        for worker in self.workers:
            worker.join(timeout=5.0)
```

## Network Monitoring Tools

### Interface Statistics Monitor

```python
# network/interface_monitor.py
import psutil
import time
from typing import Dict, Any
import threading

class InterfaceMonitor:
    def __init__(self, interface: str, update_interval: float = 1.0):
        self.interface = interface
        self.update_interval = update_interval
        self.running = False
        self.stats = {}
        self.previous_stats = {}
        
    def start_monitoring(self, callback=None):
        """Start interface monitoring"""
        self.running = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, args=(callback,))
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
    
    def _monitor_loop(self, callback):
        """Main monitoring loop"""
        while self.running:
            try:
                current_stats = self._get_interface_stats()
                
                if self.previous_stats:
                    # Calculate rates
                    time_delta = current_stats['timestamp'] - self.previous_stats['timestamp']
                    if time_delta > 0:
                        current_stats['bytes_sent_rate'] = (
                            current_stats['bytes_sent'] - self.previous_stats['bytes_sent']
                        ) / time_delta
                        current_stats['bytes_recv_rate'] = (
                            current_stats['bytes_recv'] - self.previous_stats['bytes_recv']
                        ) / time_delta
                        current_stats['packets_sent_rate'] = (
                            current_stats['packets_sent'] - self.previous_stats['packets_sent']
                        ) / time_delta
                        current_stats['packets_recv_rate'] = (
                            current_stats['packets_recv'] - self.previous_stats['packets_recv']
                        ) / time_delta
                
                self.stats = current_stats
                self.previous_stats = current_stats.copy()
                
                if callback:
                    callback(current_stats)
                
                time.sleep(self.update_interval)
                
            except Exception as e:
                print(f"Interface monitoring error: {e}")
                time.sleep(self.update_interval)
    
    def _get_interface_stats(self) -> Dict[str, Any]:
        """Get current interface statistics"""
        net_io = psutil.net_io_counters(pernic=True)
        
        if self.interface not in net_io:
            raise ValueError(f"Interface {self.interface} not found")
        
        interface_stats = net_io[self.interface]
        
        return {
            'interface': self.interface,
            'timestamp': time.time(),
            'bytes_sent': interface_stats.bytes_sent,
            'bytes_recv': interface_stats.bytes_recv,
            'packets_sent': interface_stats.packets_sent,
            'packets_recv': interface_stats.packets_recv,
            'errin': interface_stats.errin,
            'errout': interface_stats.errout,
            'dropin': interface_stats.dropin,
            'dropout': interface_stats.dropout
        }
    
    def get_current_stats(self) -> Dict[str, Any]:
        """Get current statistics"""
        return self.stats.copy()
    
    def stop_monitoring(self):
        """Stop monitoring"""
        self.running = False
        if hasattr(self, 'monitor_thread'):
            self.monitor_thread.join(timeout=5.0)
```

### Network Latency Monitor

```python
# network/latency_monitor.py
import subprocess
import threading
import time
import statistics
from typing import List, Dict, Optional

class LatencyMonitor:
    def __init__(self, targets: List[str], interval: float = 60.0):
        self.targets = targets
        self.interval = interval
        self.running = False
        self.latency_data = {}
        
    def start_monitoring(self):
        """Start latency monitoring"""
        self.running = True
        for target in self.targets:
            thread = threading.Thread(
                target=self._monitor_target,
                args=(target,),
                name=f"LatencyMonitor-{target}"
            )
            thread.daemon = True
            thread.start()
    
    def _monitor_target(self, target: str):
        """Monitor latency to specific target"""
        while self.running:
            try:
                latencies = self._ping_target(target, count=5)
                if latencies:
                    self.latency_data[target] = {
                        'timestamp': time.time(),
                        'min': min(latencies),
                        'max': max(latencies),
                        'avg': statistics.mean(latencies),
                        'median': statistics.median(latencies),
                        'packet_loss': self._calculate_packet_loss(latencies, 5)
                    }
                
                time.sleep(self.interval)
                
            except Exception as e:
                print(f"Latency monitoring error for {target}: {e}")
                time.sleep(self.interval)
    
    def _ping_target(self, target: str, count: int = 5) -> List[float]:
        """Ping target and return latencies"""
        try:
            cmd = ['ping', '-c', str(count), '-W', '2', target]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            
            if result.returncode != 0:
                return []
            
            latencies = []
            for line in result.stdout.split('\n'):
                if 'time=' in line:
                    time_part = line.split('time=')[1].split()[0]
                    latencies.append(float(time_part))
            
            return latencies
            
        except Exception as e:
            print(f"Ping error: {e}")
            return []
    
    def _calculate_packet_loss(self, latencies: List[float], sent: int) -> float:
        """Calculate packet loss percentage"""
        received = len(latencies)
        return ((sent - received) / sent) * 100
    
    def get_latency_stats(self) -> Dict[str, Dict]:
        """Get current latency statistics"""
        return self.latency_data.copy()
    
    def stop_monitoring(self):
        """Stop monitoring"""
        self.running = False
```

## Network Actions & Response

### Firewall Rule Management

```python
# network/firewall_manager.py
import subprocess
import threading
import time
from typing import List, Dict, Optional
from dataclasses import dataclass

@dataclass
class FirewallRule:
    rule_id: str
    action: str  # 'block', 'allow', 'drop'
    target_type: str  # 'ip', 'port', 'protocol'
    target_value: str
    direction: str  # 'input', 'output', 'forward'
    duration: Optional[int] = None  # seconds, None for permanent
    created_at: float = None
    expires_at: Optional[float] = None

class FirewallManager:
    def __init__(self, firewall_type: str = 'iptables'):
        self.firewall_type = firewall_type
        self.active_rules: Dict[str, FirewallRule] = {}
        self.rule_lock = threading.RLock()
        
        # Start cleanup thread
        self.cleanup_thread = threading.Thread(target=self._cleanup_expired_rules, daemon=True)
        self.cleanup_thread.start()
    
    def block_ip(self, ip_address: str, duration: Optional[int] = None, 
                 reason: str = "") -> str:
        """Block IP address"""
        rule_id = f"block_ip_{ip_address}_{int(time.time())}"
        
        try:
            # Execute iptables command
            cmd = [
                'iptables',
                '-I', 'INPUT',
                '-s', ip_address,
                '-j', 'DROP',
                '-m', 'comment',
                '--comment', f'IDS_AI_BLOCK_{rule_id}'
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            
            # Create rule record
            rule = FirewallRule(
                rule_id=rule_id,
                action='block',
                target_type='ip',
                target_value=ip_address,
                direction='input',
                duration=duration,
                created_at=time.time(),
                expires_at=time.time() + duration if duration else None
            )
            
            with self.rule_lock:
                self.active_rules[rule_id] = rule
            
            print(f"Blocked IP {ip_address} with rule {rule_id}")
            return rule_id
            
        except subprocess.CalledProcessError as e:
            print(f"Failed to block IP {ip_address}: {e.stderr}")
            raise
    
    def unblock_ip(self, ip_address: str) -> bool:
        """Unblock IP address"""
        try:
            # Find and remove all rules for this IP
            rules_to_remove = []
            with self.rule_lock:
                for rule_id, rule in self.active_rules.items():
                    if rule.target_value == ip_address and rule.action == 'block':
                        rules_to_remove.append(rule_id)
            
            success = True
            for rule_id in rules_to_remove:
                if self._remove_iptables_rule(rule_id):
                    with self.rule_lock:
                        self.active_rules.pop(rule_id, None)
                    print(f"Removed rule {rule_id} for IP {ip_address}")
                else:
                    success = False
            
            return success
            
        except Exception as e:
            print(f"Failed to unblock IP {ip_address}: {e}")
            return False
    
    def block_port(self, port: int, protocol: str = 'tcp', 
                   duration: Optional[int] = None) -> str:
        """Block port"""
        rule_id = f"block_port_{protocol}_{port}_{int(time.time())}"
        
        try:
            cmd = [
                'iptables',
                '-I', 'INPUT',
                '-p', protocol.lower(),
                '--dport', str(port),
                '-j', 'DROP',
                '-m', 'comment',
                '--comment', f'IDS_AI_BLOCK_{rule_id}'
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            
            rule = FirewallRule(
                rule_id=rule_id,
                action='block',
                target_type='port',
                target_value=f"{protocol}:{port}",
                direction='input',
                duration=duration,
                created_at=time.time(),
                expires_at=time.time() + duration if duration else None
            )
            
            with self.rule_lock:
                self.active_rules[rule_id] = rule
            
            print(f"Blocked {protocol} port {port} with rule {rule_id}")
            return rule_id
            
        except subprocess.CalledProcessError as e:
            print(f"Failed to block port {port}: {e.stderr}")
            raise
    
    def _remove_iptables_rule(self, rule_id: str) -> bool:
        """Remove iptables rule by comment"""
        try:
            # List rules with line numbers
            result = subprocess.run(
                ['iptables', '-L', 'INPUT', '--line-numbers', '-v'],
                capture_output=True, text=True, check=True
            )
            
            # Find rule line number by comment
            line_number = None
            for line in result.stdout.split('\n'):
                if f'IDS_AI_BLOCK_{rule_id}' in line:
                    line_number = line.split()[0]
                    break
            
            if line_number:
                # Remove rule by line number
                cmd = ['iptables', '-D', 'INPUT', line_number]
                subprocess.run(cmd, capture_output=True, text=True, check=True)
                return True
            
            return False
            
        except subprocess.CalledProcessError as e:
            print(f"Failed to remove iptables rule {rule_id}: {e.stderr}")
            return False
    
    def _cleanup_expired_rules(self):
        """Background thread to clean up expired rules"""
        while True:
            try:
                current_time = time.time()
                expired_rules = []
                
                with self.rule_lock:
                    for rule_id, rule in self.active_rules.items():
                        if rule.expires_at and current_time >= rule.expires_at:
                            expired_rules.append(rule_id)
                
                for rule_id in expired_rules:
                    rule = self.active_rules.get(rule_id)
                    if rule:
                        if rule.target_type == 'ip':
                            self.unblock_ip(rule.target_value)
                        elif rule.target_type == 'port':
                            protocol, port = rule.target_value.split(':')
                            self.unblock_port(int(port), protocol)
                
                time.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                print(f"Cleanup thread error: {e}")
                time.sleep(30)
    
    def get_active_rules(self) -> List[Dict]:
        """Get list of active firewall rules"""
        with self.rule_lock:
            return [
                {
                    'rule_id': rule.rule_id,
                    'action': rule.action,
                    'target_type': rule.target_type,
                    'target_value': rule.target_value,
                    'created_at': rule.created_at,
                    'expires_at': rule.expires_at,
                    'time_remaining': rule.expires_at - time.time() if rule.expires_at else None
                }
                for rule in self.active_rules.values()
            ]
    
    def get_rule_stats(self) -> Dict[str, int]:
        """Get firewall rule statistics"""
        with self.rule_lock:
            stats = {
                'total_rules': len(self.active_rules),
                'ip_blocks': 0,
                'port_blocks': 0,
                'temporary_rules': 0,
                'permanent_rules': 0
            }
            
            for rule in self.active_rules.values():
                if rule.target_type == 'ip':
                    stats['ip_blocks'] += 1
                elif rule.target_type == 'port':
                    stats['port_blocks'] += 1
                
                if rule.expires_at:
                    stats['temporary_rules'] += 1
                else:
                    stats['permanent_rules'] += 1
            
            return stats
```

### Network Quality of Service (QoS)

```python
# network/qos_manager.py
import subprocess
from typing import Dict, List
from dataclasses import dataclass

@dataclass
class QoSRule:
    rule_id: str
    interface: str
    source_ip: str
    bandwidth_limit: int  # in kbps
    priority: int  # 1-7, higher is better
    
class QoSManager:
    def __init__(self):
        self.active_rules: Dict[str, QoSRule] = {}
    
    def limit_bandwidth(self, source_ip: str, interface: str, 
                       limit_kbps: int, priority: int = 3) -> str:
        """Limit bandwidth for specific IP"""
        rule_id = f"qos_{source_ip}_{interface}_{int(time.time())}"
        
        try:
            # Create traffic control rules
            self._setup_qdisc(interface)
            self._add_class(interface, rule_id, limit_kbps, priority)
            self._add_filter(interface, source_ip, rule_id)
            
            rule = QoSRule(
                rule_id=rule_id,
                interface=interface,
                source_ip=source_ip,
                bandwidth_limit=limit_kbps,
                priority=priority
            )
            
            self.active_rules[rule_id] = rule
            print(f"Applied QoS rule {rule_id} for {source_ip}")
            return rule_id
            
        except Exception as e:
            print(f"Failed to apply QoS rule: {e}")
            raise
    
    def _setup_qdisc(self, interface: str):
        """Setup traffic control queuing discipline"""
        try:
            # Remove existing qdisc
            subprocess.run(['tc', 'qdisc', 'del', 'dev', interface, 'root'], 
                         capture_output=True)
            
            # Add HTB qdisc
            cmd = ['tc', 'qdisc', 'add', 'dev', interface, 'root', 'handle', '1:', 'htb']
            subprocess.run(cmd, capture_output=True, text=True, check=True)
            
        except subprocess.CalledProcessError as e:
            if "RTNETLINK answers: No such file or directory" not in e.stderr:
                raise
    
    def _add_class(self, interface: str, rule_id: str, rate_kbps: int, priority: int):
        """Add traffic class"""
        class_id = f"1:{abs(hash(rule_id)) % 1000 + 100}"
        
        cmd = [
            'tc', 'class', 'add', 'dev', interface,
            'parent', '1:', 'classid', class_id,
            'htb', 'rate', f'{rate_kbps}kbit',
            'ceil', f'{rate_kbps * 2}kbit',
            'prio', str(priority)
        ]
        
        subprocess.run(cmd, capture_output=True, text=True, check=True)
    
    def _add_filter(self, interface: str, source_ip: str, rule_id: str):
        """Add traffic filter"""
        class_id = f"1:{abs(hash(rule_id)) % 1000 + 100}"
        
        cmd = [
            'tc', 'filter', 'add', 'dev', interface,
            'parent', '1:', 'protocol', 'ip',
            'prio', '1', 'u32',
            'match', 'ip', 'src', source_ip,
            'flowid', class_id
        ]
        
        subprocess.run(cmd, capture_output=True, text=True, check=True)
```

This comprehensive network track documentation provides detailed implementation guidance for all network-related components of the IDS-AI system, including traffic capture, flow analysis, packet processing, monitoring tools, and network response capabilities.
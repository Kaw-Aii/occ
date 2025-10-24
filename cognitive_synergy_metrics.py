#!/usr/bin/env python3
"""
Cognitive Synergy Metrics - Measurement and Analysis
===================================================

This module provides comprehensive metrics for measuring and analyzing
cognitive synergy effectiveness in the OpenCog Collection.

Implements metrics based on:
- Goertzel's formal model of cognitive synergy
- Information theory measures
- Graph-theoretic analysis
- Performance benchmarking

Author: OpenCog Collection Contributors
License: GPL-3.0+
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass, field
from datetime import datetime
import json
import logging
from collections import defaultdict, deque

logger = logging.getLogger(__name__)


@dataclass
class SynergyMetrics:
    """Container for cognitive synergy metrics."""
    timestamp: datetime = field(default_factory=datetime.now)
    
    # Process-level metrics
    process_efficiency: float = 0.0
    bottleneck_resolution_rate: float = 0.0
    inter_process_communication: float = 0.0
    
    # Memory-level metrics
    attention_distribution_entropy: float = 0.0
    pattern_diversity: float = 0.0
    memory_coherence: float = 0.0
    
    # Synergy-specific metrics
    emergent_capability_index: float = 0.0
    cross_paradigm_synergy: float = 0.0
    bottleneck_assistance_effectiveness: float = 0.0
    
    # Performance metrics
    throughput: float = 0.0
    latency: float = 0.0
    resource_utilization: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "process_efficiency": self.process_efficiency,
            "bottleneck_resolution_rate": self.bottleneck_resolution_rate,
            "inter_process_communication": self.inter_process_communication,
            "attention_distribution_entropy": self.attention_distribution_entropy,
            "pattern_diversity": self.pattern_diversity,
            "memory_coherence": self.memory_coherence,
            "emergent_capability_index": self.emergent_capability_index,
            "cross_paradigm_synergy": self.cross_paradigm_synergy,
            "bottleneck_assistance_effectiveness": self.bottleneck_assistance_effectiveness,
            "throughput": self.throughput,
            "latency": self.latency,
            "resource_utilization": self.resource_utilization
        }


class CognitiveSynergyMetricsCollector:
    """
    Collects and analyzes cognitive synergy metrics over time.
    """
    
    def __init__(self, window_size: int = 100):
        """
        Initialize metrics collector.
        
        Args:
            window_size: Number of recent measurements to keep for analysis
        """
        self.window_size = window_size
        self.metrics_history = deque(maxlen=window_size)
        self.event_log = []
        
        # Tracking variables
        self.bottleneck_events = []
        self.synergy_events = []
        self.process_interactions = defaultdict(int)
        
    def collect_metrics(self, memory, processes: Dict[str, Any]) -> SynergyMetrics:
        """
        Collect current cognitive synergy metrics.
        
        Args:
            memory: Hypergraph memory instance
            processes: Dictionary of active cognitive processes
            
        Returns:
            Current synergy metrics
        """
        metrics = SynergyMetrics()
        
        # Process-level metrics
        metrics.process_efficiency = self._calculate_process_efficiency(processes)
        metrics.bottleneck_resolution_rate = self._calculate_bottleneck_resolution_rate()
        metrics.inter_process_communication = self._calculate_inter_process_communication()
        
        # Memory-level metrics
        metrics.attention_distribution_entropy = self._calculate_attention_entropy(memory)
        metrics.pattern_diversity = self._calculate_pattern_diversity(memory)
        metrics.memory_coherence = self._calculate_memory_coherence(memory)
        
        # Synergy-specific metrics
        metrics.emergent_capability_index = self._calculate_emergent_capability_index()
        metrics.cross_paradigm_synergy = self._calculate_cross_paradigm_synergy(processes)
        metrics.bottleneck_assistance_effectiveness = self._calculate_assistance_effectiveness()
        
        # Performance metrics
        metrics.throughput = self._calculate_throughput()
        metrics.latency = self._calculate_latency()
        metrics.resource_utilization = self._calculate_resource_utilization(memory, processes)
        
        # Store in history
        self.metrics_history.append(metrics)
        
        return metrics
    
    def _calculate_process_efficiency(self, processes: Dict[str, Any]) -> float:
        """Calculate overall process efficiency."""
        if not processes:
            return 0.0
        
        active_count = sum(1 for p in processes.values() 
                          if not (p.is_stuck if hasattr(p, 'is_stuck') else p.get('is_stuck', False)))
        total_count = len(processes)
        
        return active_count / total_count if total_count > 0 else 0.0
    
    def _calculate_bottleneck_resolution_rate(self) -> float:
        """Calculate rate of successful bottleneck resolutions."""
        if not self.bottleneck_events:
            return 1.0  # No bottlenecks is good
        
        recent_events = [e for e in self.bottleneck_events 
                        if (datetime.now() - e['timestamp']).seconds < 300]  # Last 5 minutes
        
        if not recent_events:
            return 1.0
        
        resolved = sum(1 for e in recent_events if e.get('resolved', False))
        return resolved / len(recent_events)
    
    def _calculate_inter_process_communication(self) -> float:
        """Calculate inter-process communication effectiveness."""
        if not self.process_interactions:
            return 0.0
        
        total_interactions = sum(self.process_interactions.values())
        unique_pairs = len(self.process_interactions)
        
        # Normalize by potential maximum interactions
        return min(1.0, (unique_pairs / 10.0) * (total_interactions / 100.0))
    
    def _calculate_attention_entropy(self, memory) -> float:
        """Calculate entropy of attention distribution (information-theoretic)."""
        if not hasattr(memory, 'attention_bank'):
            return 0.0
        
        attention_values = list(memory.attention_bank.values())
        if not attention_values:
            return 0.0
        
        # Normalize to probabilities
        total_attention = sum(attention_values)
        if total_attention == 0:
            return 0.0
        
        probabilities = [a / total_attention for a in attention_values]
        
        # Calculate Shannon entropy
        entropy = -sum(p * np.log2(p) if p > 0 else 0 for p in probabilities)
        
        # Normalize by maximum possible entropy
        max_entropy = np.log2(len(attention_values)) if len(attention_values) > 1 else 1.0
        
        return entropy / max_entropy if max_entropy > 0 else 0.0
    
    def _calculate_pattern_diversity(self, memory) -> float:
        """Calculate diversity of discovered patterns."""
        if not hasattr(memory, 'pattern_cache'):
            return 0.0
        
        pattern_count = len(memory.pattern_cache)
        
        # Normalize by a reasonable maximum
        return min(1.0, pattern_count / 50.0)
    
    def _calculate_memory_coherence(self, memory) -> float:
        """Calculate coherence of hypergraph memory structure."""
        if not hasattr(memory, 'atoms'):
            return 0.0
        
        atom_count = len(memory.atoms)
        if atom_count == 0:
            return 0.0
        
        # Calculate average connectivity
        total_connections = 0
        for atom in memory.atoms.values():
            total_connections += len(atom.incoming) + len(atom.outgoing)
        
        avg_connectivity = total_connections / (atom_count * 2) if atom_count > 0 else 0.0
        
        # Normalize (assuming average connectivity of 5 is good)
        return min(1.0, avg_connectivity / 5.0)
    
    def _calculate_emergent_capability_index(self) -> float:
        """Calculate index of emergent capabilities from synergy."""
        if not self.synergy_events:
            return 0.0
        
        recent_synergies = [e for e in self.synergy_events
                           if (datetime.now() - e['timestamp']).seconds < 600]  # Last 10 minutes
        
        # Weight by novelty and effectiveness
        emergence_score = sum(e.get('novelty', 0.5) * e.get('effectiveness', 0.5)
                             for e in recent_synergies)
        
        return min(1.0, emergence_score / 10.0)
    
    def _calculate_cross_paradigm_synergy(self, processes: Dict[str, Any]) -> float:
        """Calculate synergy between different AI paradigms."""
        if not processes:
            return 0.0
        
        # Identify different paradigms
        paradigms = set(
            p.process_type if hasattr(p, 'process_type') else p.get('process_type', 'unknown')
            for p in processes.values()
        )
        
        # More paradigms = more potential synergy
        paradigm_diversity = len(paradigms) / 5.0  # Normalize by expected max
        
        # Check for actual inter-paradigm interactions
        inter_paradigm_interactions = sum(
            1 for (p1, p2) in self.process_interactions.keys()
            if self._get_process_type(processes.get(p1)) != 
               self._get_process_type(processes.get(p2))
        )
        
        interaction_score = min(1.0, inter_paradigm_interactions / 20.0)
        
        return (paradigm_diversity + interaction_score) / 2.0
    
    def _get_process_type(self, process) -> str:
        """Helper to get process type from either object or dict."""
        if process is None:
            return 'unknown'
        if hasattr(process, 'process_type'):
            return process.process_type
        if isinstance(process, dict):
            return process.get('process_type', 'unknown')
        return 'unknown'
    
    def _calculate_assistance_effectiveness(self) -> float:
        """Calculate effectiveness of bottleneck assistance."""
        if not self.bottleneck_events:
            return 1.0  # No bottlenecks = perfect
        
        recent_assisted = [e for e in self.bottleneck_events
                          if e.get('assistance_provided', False) and
                          (datetime.now() - e['timestamp']).seconds < 300]
        
        if not recent_assisted:
            return 0.5  # Neutral if no recent assistance
        
        successful = sum(1 for e in recent_assisted if e.get('resolved', False))
        return successful / len(recent_assisted)
    
    def _calculate_throughput(self) -> float:
        """Calculate cognitive processing throughput."""
        if len(self.metrics_history) < 2:
            return 0.0
        
        # Estimate based on recent activity
        recent_events = len(self.event_log[-100:])  # Last 100 events
        time_window = 60.0  # 1 minute
        
        return recent_events / time_window
    
    def _calculate_latency(self) -> float:
        """Calculate average cognitive processing latency."""
        if not self.event_log:
            return 0.0
        
        recent_events = self.event_log[-50:]
        latencies = [e.get('latency', 0.0) for e in recent_events if 'latency' in e]
        
        return np.mean(latencies) if latencies else 0.0
    
    def _calculate_resource_utilization(self, memory, processes: Dict[str, Any]) -> float:
        """Calculate resource utilization efficiency."""
        # Combine memory and process utilization
        memory_util = len(memory.atoms) / 10000.0 if hasattr(memory, 'atoms') else 0.0
        process_util = len(processes) / 10.0 if processes else 0.0
        
        return min(1.0, (memory_util + process_util) / 2.0)
    
    def record_bottleneck_event(self, process_id: str, resolved: bool = False,
                               assistance_provided: bool = False):
        """Record a bottleneck event."""
        event = {
            'timestamp': datetime.now(),
            'process_id': process_id,
            'resolved': resolved,
            'assistance_provided': assistance_provided
        }
        self.bottleneck_events.append(event)
        self.event_log.append(event)
    
    def record_synergy_event(self, process_ids: List[str], novelty: float = 0.5,
                            effectiveness: float = 0.5):
        """Record a cognitive synergy event."""
        event = {
            'timestamp': datetime.now(),
            'process_ids': process_ids,
            'novelty': novelty,
            'effectiveness': effectiveness
        }
        self.synergy_events.append(event)
        self.event_log.append(event)
        
        # Track process interactions
        for i, p1 in enumerate(process_ids):
            for p2 in process_ids[i+1:]:
                self.process_interactions[(p1, p2)] += 1
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get summary of recent metrics."""
        if not self.metrics_history:
            return {}
        
        recent_metrics = list(self.metrics_history)[-10:]  # Last 10 measurements
        
        summary = {
            'current': recent_metrics[-1].to_dict() if recent_metrics else {},
            'averages': {},
            'trends': {}
        }
        
        # Calculate averages
        for key in ['process_efficiency', 'attention_distribution_entropy',
                   'pattern_diversity', 'emergent_capability_index']:
            values = [getattr(m, key) for m in recent_metrics]
            summary['averages'][key] = np.mean(values) if values else 0.0
            
            # Calculate trend (positive = improving)
            if len(values) >= 2:
                trend = values[-1] - values[0]
                summary['trends'][key] = 'improving' if trend > 0.01 else \
                                        'declining' if trend < -0.01 else 'stable'
        
        return summary
    
    def export_metrics(self, filename: str):
        """Export metrics history to file."""
        metrics_data = [m.to_dict() for m in self.metrics_history]
        
        with open(filename, 'w') as f:
            json.dump({
                'metrics': metrics_data,
                'summary': self.get_metrics_summary()
            }, f, indent=2)
        
        logger.info(f"Exported metrics to {filename}")


def visualize_metrics(metrics_collector: CognitiveSynergyMetricsCollector):
    """
    Create a simple text-based visualization of metrics.
    """
    summary = metrics_collector.get_metrics_summary()
    
    print("\n" + "="*60)
    print("COGNITIVE SYNERGY METRICS DASHBOARD")
    print("="*60)
    
    if 'current' in summary and summary['current']:
        current = summary['current']
        print("\nCurrent Metrics:")
        print(f"  Process Efficiency:        {current.get('process_efficiency', 0):.2%}")
        print(f"  Attention Entropy:         {current.get('attention_distribution_entropy', 0):.2%}")
        print(f"  Pattern Diversity:         {current.get('pattern_diversity', 0):.2%}")
        print(f"  Emergent Capability Index: {current.get('emergent_capability_index', 0):.2%}")
        print(f"  Cross-Paradigm Synergy:    {current.get('cross_paradigm_synergy', 0):.2%}")
    
    if 'trends' in summary and summary['trends']:
        print("\nTrends:")
        for metric, trend in summary['trends'].items():
            symbol = "↑" if trend == "improving" else "↓" if trend == "declining" else "→"
            print(f"  {metric:30s} {symbol} {trend}")
    
    print("\n" + "="*60 + "\n")


if __name__ == "__main__":
    # Test metrics collection
    logging.basicConfig(level=logging.INFO)
    
    print("Testing Cognitive Synergy Metrics...")
    
    # Create mock memory and processes
    class MockMemory:
        def __init__(self):
            self.atoms = {}
            self.attention_bank = {'atom1': 0.5, 'atom2': 0.3, 'atom3': 0.2}
            self.pattern_cache = {'pattern1': [], 'pattern2': []}
    
    memory = MockMemory()
    processes = {
        'reasoning': {'process_type': 'symbolic', 'is_stuck': False},
        'learning': {'process_type': 'ml', 'is_stuck': False}
    }
    
    collector = CognitiveSynergyMetricsCollector()
    
    # Collect some metrics
    for i in range(5):
        metrics = collector.collect_metrics(memory, processes)
        collector.record_synergy_event(['reasoning', 'learning'], novelty=0.7, effectiveness=0.8)
    
    # Visualize
    visualize_metrics(collector)
    
    # Export
    collector.export_metrics('cognitive_synergy_metrics.json')
    print("Metrics exported to cognitive_synergy_metrics.json")


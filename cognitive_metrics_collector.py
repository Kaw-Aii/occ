"""
Cognitive Metrics Collector
============================

Real-time metrics collection and analysis for cognitive synergy monitoring.
Tracks system health, component performance, and emergent synergy properties.

Author: OpenCog Collection Contributors
License: GPL-3.0+
"""

import time
import json
import os
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field, asdict
from datetime import datetime
from collections import deque
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class CognitiveMetrics:
    """Container for cognitive system metrics."""
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    # System-level metrics
    system_health: float = 1.0
    uptime_seconds: float = 0.0
    
    # AAR metrics
    agent_action_potential: float = 0.0
    arena_knowledge_density: float = 0.0
    arena_coherence: float = 1.0
    relation_self_coherence: float = 1.0
    
    # Hypergraph metrics
    hypergraph_node_count: int = 0
    hypergraph_edge_count: int = 0
    hypergraph_connectivity: float = 0.0
    
    # Attention metrics
    attention_focus_entropy: float = 0.0
    attention_allocation_efficiency: float = 1.0
    
    # Neural-symbolic metrics
    neural_symbolic_translation_accuracy: float = 0.0
    symbolic_neural_translation_accuracy: float = 0.0
    
    # Multi-agent metrics
    agent_count: int = 0
    collaboration_efficiency: float = 0.0
    message_passing_rate: float = 0.0
    
    # Synergy metrics
    cognitive_synergy_score: float = 0.0
    emergence_index: float = 0.0
    integration_level: float = 0.0
    
    # Performance metrics
    average_response_time_ms: float = 0.0
    throughput_ops_per_sec: float = 0.0
    memory_usage_mb: float = 0.0
    
    # Component status
    component_status: Dict[str, str] = field(default_factory=dict)
    active_components: int = 0


class CognitiveMetricsCollector:
    """
    Collects, aggregates, and analyzes cognitive system metrics.
    Provides real-time monitoring and historical analysis capabilities.
    """
    
    def __init__(self, history_size: int = 1000):
        self.history_size = history_size
        self.metrics_history: deque = deque(maxlen=history_size)
        self.start_time = time.time()
        self.operation_count = 0
        self.total_response_time = 0.0
        
        logger.info("CognitiveMetricsCollector initialized")
    
    def collect_metrics(self) -> CognitiveMetrics:
        """
        Collect current metrics from all cognitive components.
        Returns a CognitiveMetrics object with current values.
        """
        metrics = CognitiveMetrics()
        
        # System metrics
        metrics.uptime_seconds = time.time() - self.start_time
        metrics.system_health = self._compute_system_health()
        
        # Collect from components
        metrics = self._collect_aar_metrics(metrics)
        metrics = self._collect_hypergraph_metrics(metrics)
        metrics = self._collect_attention_metrics(metrics)
        metrics = self._collect_neural_symbolic_metrics(metrics)
        metrics = self._collect_multi_agent_metrics(metrics)
        
        # Compute derived metrics
        metrics.cognitive_synergy_score = self._compute_synergy_score(metrics)
        metrics.emergence_index = self._compute_emergence_index(metrics)
        metrics.integration_level = self._compute_integration_level(metrics)
        
        # Performance metrics
        if self.operation_count > 0:
            metrics.average_response_time_ms = (
                self.total_response_time / self.operation_count * 1000
            )
        metrics.throughput_ops_per_sec = (
            self.operation_count / metrics.uptime_seconds 
            if metrics.uptime_seconds > 0 else 0
        )
        
        # Store in history
        self.metrics_history.append(metrics)
        
        return metrics
    
    def _compute_system_health(self) -> float:
        """Compute overall system health score (0-1)."""
        # Simple heuristic: healthy if no recent errors
        # In production, this would check error rates, resource usage, etc.
        return 1.0
    
    def _collect_aar_metrics(self, metrics: CognitiveMetrics) -> CognitiveMetrics:
        """Collect metrics from AAR components."""
        try:
            from self_awareness_aar import AgentState, ArenaState, RelationState
            
            # In production, these would be singleton instances
            # For now, we check if components are available
            metrics.component_status['aar_agent'] = 'available'
            metrics.component_status['aar_arena'] = 'available'
            metrics.component_status['aar_relation'] = 'available'
            metrics.active_components += 3
            
            # Would collect actual values from running instances
            metrics.agent_action_potential = 0.5  # Placeholder
            metrics.arena_knowledge_density = 0.7  # Placeholder
            metrics.arena_coherence = 0.85  # Placeholder
            metrics.relation_self_coherence = 0.9  # Placeholder
            
        except ImportError:
            metrics.component_status['aar'] = 'unavailable'
        except Exception as e:
            logger.warning(f"Failed to collect AAR metrics: {e}")
            metrics.component_status['aar'] = 'error'
        
        return metrics
    
    def _collect_hypergraph_metrics(self, metrics: CognitiveMetrics) -> CognitiveMetrics:
        """Collect metrics from hypergraph."""
        try:
            from hypergraph_dynamics import HypergraphDynamics
            
            metrics.component_status['hypergraph'] = 'available'
            metrics.active_components += 1
            
            # Would collect actual values from running instance
            metrics.hypergraph_node_count = 100  # Placeholder
            metrics.hypergraph_edge_count = 250  # Placeholder
            
            if metrics.hypergraph_node_count > 0:
                metrics.hypergraph_connectivity = (
                    metrics.hypergraph_edge_count / metrics.hypergraph_node_count
                )
            
        except ImportError:
            metrics.component_status['hypergraph'] = 'unavailable'
        except Exception as e:
            logger.warning(f"Failed to collect hypergraph metrics: {e}")
            metrics.component_status['hypergraph'] = 'error'
        
        return metrics
    
    def _collect_attention_metrics(self, metrics: CognitiveMetrics) -> CognitiveMetrics:
        """Collect metrics from attention allocation."""
        try:
            from attention_allocation import AttentionAllocator
            
            metrics.component_status['attention'] = 'available'
            metrics.active_components += 1
            
            # Would collect actual values from running instance
            metrics.attention_focus_entropy = 0.6  # Placeholder
            metrics.attention_allocation_efficiency = 0.8  # Placeholder
            
        except ImportError:
            metrics.component_status['attention'] = 'unavailable'
        except Exception as e:
            logger.warning(f"Failed to collect attention metrics: {e}")
            metrics.component_status['attention'] = 'error'
        
        return metrics
    
    def _collect_neural_symbolic_metrics(self, metrics: CognitiveMetrics) -> CognitiveMetrics:
        """Collect metrics from neural-symbolic integration."""
        try:
            from neural_symbolic_integration import NeuralSymbolicIntegration
            
            metrics.component_status['neural_symbolic'] = 'available'
            metrics.active_components += 1
            
            # Would collect actual values from running instance
            metrics.neural_symbolic_translation_accuracy = 0.75  # Placeholder
            metrics.symbolic_neural_translation_accuracy = 0.80  # Placeholder
            
        except ImportError:
            metrics.component_status['neural_symbolic'] = 'unavailable'
        except Exception as e:
            logger.warning(f"Failed to collect neural-symbolic metrics: {e}")
            metrics.component_status['neural_symbolic'] = 'error'
        
        return metrics
    
    def _collect_multi_agent_metrics(self, metrics: CognitiveMetrics) -> CognitiveMetrics:
        """Collect metrics from multi-agent system."""
        try:
            from multi_agent_collaboration import MultiAgentCollaboration
            
            metrics.component_status['multi_agent'] = 'available'
            metrics.active_components += 1
            
            # Would collect actual values from running instance
            metrics.agent_count = 5  # Placeholder
            metrics.collaboration_efficiency = 0.7  # Placeholder
            metrics.message_passing_rate = 10.0  # Placeholder (msgs/sec)
            
        except ImportError:
            metrics.component_status['multi_agent'] = 'unavailable'
        except Exception as e:
            logger.warning(f"Failed to collect multi-agent metrics: {e}")
            metrics.component_status['multi_agent'] = 'error'
        
        return metrics
    
    def _compute_synergy_score(self, metrics: CognitiveMetrics) -> float:
        """
        Compute cognitive synergy score based on component interactions.
        Higher score indicates stronger emergent capabilities.
        """
        # Weighted combination of key metrics
        score = (
            metrics.arena_coherence * 0.2 +
            metrics.relation_self_coherence * 0.2 +
            metrics.hypergraph_connectivity * 0.1 +
            metrics.attention_allocation_efficiency * 0.15 +
            metrics.collaboration_efficiency * 0.15 +
            (metrics.neural_symbolic_translation_accuracy + 
             metrics.symbolic_neural_translation_accuracy) / 2 * 0.2
        )
        
        return max(0.0, min(1.0, score))
    
    def _compute_emergence_index(self, metrics: CognitiveMetrics) -> float:
        """
        Compute emergence index - measures how much system capabilities
        exceed the sum of individual components.
        """
        # Simplified heuristic: emergence increases with integration
        # and number of active components
        if metrics.active_components == 0:
            return 0.0
        
        base_emergence = metrics.active_components / 10.0  # Normalized
        synergy_boost = metrics.cognitive_synergy_score * 0.5
        
        return max(0.0, min(1.0, base_emergence + synergy_boost))
    
    def _compute_integration_level(self, metrics: CognitiveMetrics) -> float:
        """
        Compute integration level - measures how well components
        are working together.
        """
        # Based on component availability and interaction quality
        if metrics.active_components == 0:
            return 0.0
        
        # More active components = higher potential integration
        component_factor = metrics.active_components / 8.0  # Assume 8 core components
        
        # Quality of integration based on coherence and efficiency
        quality_factor = (
            metrics.arena_coherence * 0.4 +
            metrics.attention_allocation_efficiency * 0.3 +
            metrics.collaboration_efficiency * 0.3
        )
        
        return max(0.0, min(1.0, component_factor * quality_factor))
    
    def record_operation(self, response_time: float):
        """Record an operation for performance tracking."""
        self.operation_count += 1
        self.total_response_time += response_time
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get a summary of current metrics."""
        if not self.metrics_history:
            return {}
        
        current = self.metrics_history[-1]
        
        return {
            'timestamp': current.timestamp,
            'system_health': current.system_health,
            'uptime_seconds': current.uptime_seconds,
            'cognitive_synergy_score': current.cognitive_synergy_score,
            'emergence_index': current.emergence_index,
            'integration_level': current.integration_level,
            'active_components': current.active_components,
            'throughput_ops_per_sec': current.throughput_ops_per_sec,
            'component_status': current.component_status
        }
    
    def get_historical_metrics(self, metric_name: str, count: int = 100) -> List[float]:
        """Get historical values for a specific metric."""
        values = []
        for metrics in list(self.metrics_history)[-count:]:
            if hasattr(metrics, metric_name):
                values.append(getattr(metrics, metric_name))
        return values
    
    def export_metrics(self, filepath: str):
        """Export metrics history to JSON file."""
        try:
            data = [asdict(m) for m in self.metrics_history]
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2)
            logger.info(f"Exported {len(data)} metric records to {filepath}")
        except Exception as e:
            logger.error(f"Failed to export metrics: {e}")
    
    def print_summary(self):
        """Print a formatted summary of current metrics."""
        summary = self.get_metrics_summary()
        
        print("\n" + "=" * 60)
        print("Cognitive System Metrics Summary")
        print("=" * 60)
        print(f"Timestamp: {summary.get('timestamp', 'N/A')}")
        print(f"Uptime: {summary.get('uptime_seconds', 0):.1f} seconds")
        print(f"System Health: {summary.get('system_health', 0):.2%}")
        print()
        print("Synergy Metrics:")
        print(f"  Cognitive Synergy Score: {summary.get('cognitive_synergy_score', 0):.2%}")
        print(f"  Emergence Index: {summary.get('emergence_index', 0):.2%}")
        print(f"  Integration Level: {summary.get('integration_level', 0):.2%}")
        print()
        print(f"Active Components: {summary.get('active_components', 0)}")
        print(f"Throughput: {summary.get('throughput_ops_per_sec', 0):.2f} ops/sec")
        print()
        print("Component Status:")
        for component, status in summary.get('component_status', {}).items():
            status_symbol = "✓" if status == "available" else "✗"
            print(f"  {status_symbol} {component}: {status}")
        print("=" * 60)


def main():
    """Main entry point for metrics collection demo."""
    print("Cognitive Metrics Collector - Demo")
    print("=" * 60)
    
    collector = CognitiveMetricsCollector()
    
    # Collect metrics
    print("\nCollecting metrics...")
    metrics = collector.collect_metrics()
    
    # Simulate some operations
    for i in range(10):
        time.sleep(0.01)
        collector.record_operation(0.01)
    
    # Collect again
    metrics = collector.collect_metrics()
    
    # Print summary
    collector.print_summary()
    
    # Export metrics
    export_path = "cognitive_metrics_export.json"
    collector.export_metrics(export_path)
    print(f"\n✓ Metrics exported to {export_path}")


if __name__ == "__main__":
    main()

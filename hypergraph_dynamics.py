"""
Hypergraph Dynamics Module for OpenCog Collection
=================================================

This module implements hypergraph dynamics for cognitive synergy, providing
mechanisms for pattern propagation, attention spreading, and emergent structure
formation in the cognitive architecture.

Key Features:
- Dynamic hypergraph evolution and pattern formation
- Attention spreading and activation propagation
- Structural analysis and metric computation
- Integration with AAR (Agent-Arena-Relation) architecture
- Support for cognitive synergy through distributed pattern recognition

Based on principles from:
- Hypergraph theory and algebraic topology
- Attention allocation mechanisms (ECAN)
- Pattern mining and recognition
- Cognitive synergy theory (Goertzel)

Author: OpenCog Collection Contributors
License: GPL-3.0+
"""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Set, Callable
from dataclasses import dataclass, field
from collections import defaultdict, deque
import logging
from datetime import datetime
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class HyperEdge:
    """
    Represents a hyperedge connecting multiple nodes in the hypergraph.
    Unlike regular edges, hyperedges can connect any number of nodes.
    """
    edge_id: str
    nodes: Set[str]
    edge_type: str
    weight: float = 1.0
    truth_value: float = 1.0
    attention_value: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    
    def __hash__(self):
        return hash(self.edge_id)
    
    def __eq__(self, other):
        return isinstance(other, HyperEdge) and self.edge_id == other.edge_id


@dataclass
class HyperNode:
    """
    Represents a node in the hypergraph with associated properties.
    """
    node_id: str
    node_type: str
    activation: float = 0.0
    attention: float = 0.0
    importance: float = 0.0
    incident_edges: Set[str] = field(default_factory=set)
    properties: Dict[str, Any] = field(default_factory=dict)
    
    def __hash__(self):
        return hash(self.node_id)
    
    def __eq__(self, other):
        return isinstance(other, HyperNode) and self.node_id == other.node_id


class HypergraphDynamics:
    """
    Implements dynamic hypergraph operations for cognitive synergy.
    
    Manages the evolution of hypergraph structures through:
    - Attention spreading and activation propagation
    - Pattern formation and recognition
    - Structural metrics and analysis
    - Integration with cognitive processes
    """
    
    def __init__(self, attention_decay: float = 0.1, 
                 activation_threshold: float = 0.3,
                 max_spread_depth: int = 5):
        """
        Initialize hypergraph dynamics system.
        
        Args:
            attention_decay: Rate of attention decay per time step
            activation_threshold: Minimum activation for pattern recognition
            max_spread_depth: Maximum depth for attention spreading
        """
        self.nodes: Dict[str, HyperNode] = {}
        self.edges: Dict[str, HyperEdge] = {}
        
        self.attention_decay = attention_decay
        self.activation_threshold = activation_threshold
        self.max_spread_depth = max_spread_depth
        
        # Metrics tracking
        self.metrics = {
            'total_nodes': 0,
            'total_edges': 0,
            'average_degree': 0.0,
            'clustering_coefficient': 0.0,
            'pattern_count': 0,
            'attention_entropy': 0.0
        }
        
        # Pattern cache for cognitive synergy
        self.pattern_cache: Dict[str, Any] = {}
        self.activation_history: deque = deque(maxlen=1000)
        
        logger.info("Hypergraph Dynamics initialized")
    
    def add_node(self, node_id: str, node_type: str, 
                 properties: Optional[Dict[str, Any]] = None) -> HyperNode:
        """Add a node to the hypergraph."""
        if node_id in self.nodes:
            logger.warning(f"Node {node_id} already exists, updating properties")
            node = self.nodes[node_id]
            if properties:
                node.properties.update(properties)
            return node
        
        node = HyperNode(
            node_id=node_id,
            node_type=node_type,
            properties=properties or {}
        )
        self.nodes[node_id] = node
        self.metrics['total_nodes'] = len(self.nodes)
        
        logger.debug(f"Added node: {node_id} (type: {node_type})")
        return node
    
    def add_edge(self, edge_id: str, nodes: Set[str], edge_type: str,
                 weight: float = 1.0, truth_value: float = 1.0) -> HyperEdge:
        """Add a hyperedge connecting multiple nodes."""
        # Ensure all nodes exist
        for node_id in nodes:
            if node_id not in self.nodes:
                logger.warning(f"Node {node_id} not found, creating placeholder")
                self.add_node(node_id, "unknown")
        
        edge = HyperEdge(
            edge_id=edge_id,
            nodes=nodes,
            edge_type=edge_type,
            weight=weight,
            truth_value=truth_value
        )
        
        self.edges[edge_id] = edge
        
        # Update incident edges for nodes
        for node_id in nodes:
            self.nodes[node_id].incident_edges.add(edge_id)
        
        self.metrics['total_edges'] = len(self.edges)
        logger.debug(f"Added edge: {edge_id} connecting {len(nodes)} nodes")
        
        return edge
    
    def spread_attention(self, source_nodes: List[str], 
                        initial_attention: float = 1.0,
                        spread_factor: float = 0.7) -> Dict[str, float]:
        """
        Spread attention from source nodes through the hypergraph.
        
        Implements attention spreading mechanism similar to ECAN
        (Economic Attention Networks) for cognitive synergy.
        
        Args:
            source_nodes: Starting nodes for attention spreading
            initial_attention: Initial attention value
            spread_factor: Decay factor for attention spreading
            
        Returns:
            Dictionary mapping node_id to attention value
        """
        attention_map: Dict[str, float] = defaultdict(float)
        visited: Set[str] = set()
        queue: deque = deque()
        
        # Initialize with source nodes
        for node_id in source_nodes:
            if node_id in self.nodes:
                queue.append((node_id, initial_attention, 0))
                attention_map[node_id] = initial_attention
        
        # Breadth-first spreading
        while queue:
            current_node, current_attention, depth = queue.popleft()
            
            if depth >= self.max_spread_depth:
                continue
            
            if current_node in visited:
                continue
            visited.add(current_node)
            
            # Update node attention
            node = self.nodes[current_node]
            node.attention = max(node.attention, current_attention)
            
            # Spread to neighbors through hyperedges
            for edge_id in node.incident_edges:
                edge = self.edges[edge_id]
                
                # Calculate attention to spread based on edge properties
                spread_attention = current_attention * spread_factor * edge.weight
                
                if spread_attention < self.activation_threshold:
                    continue
                
                # Spread to all nodes in the hyperedge
                for neighbor_id in edge.nodes:
                    if neighbor_id != current_node and neighbor_id not in visited:
                        new_attention = attention_map[neighbor_id] + spread_attention
                        attention_map[neighbor_id] = new_attention
                        queue.append((neighbor_id, new_attention, depth + 1))
        
        logger.info(f"Attention spread to {len(attention_map)} nodes from {len(source_nodes)} sources")
        return dict(attention_map)
    
    def activate_pattern(self, pattern_nodes: List[str], 
                        activation_strength: float = 1.0) -> float:
        """
        Activate a pattern in the hypergraph and measure resonance.
        
        Returns the pattern coherence score (0-1).
        """
        if not pattern_nodes:
            return 0.0
        
        # Set activation for pattern nodes
        for node_id in pattern_nodes:
            if node_id in self.nodes:
                self.nodes[node_id].activation = activation_strength
        
        # Measure pattern coherence through edge connectivity
        pattern_set = set(pattern_nodes)
        connecting_edges = 0
        total_possible = len(pattern_nodes) * (len(pattern_nodes) - 1) / 2
        
        for edge in self.edges.values():
            if len(edge.nodes.intersection(pattern_set)) >= 2:
                connecting_edges += 1
        
        coherence = connecting_edges / max(total_possible, 1.0)
        
        # Record activation
        self.activation_history.append({
            'timestamp': datetime.now().isoformat(),
            'pattern_size': len(pattern_nodes),
            'coherence': coherence,
            'activation_strength': activation_strength
        })
        
        logger.debug(f"Pattern activated: {len(pattern_nodes)} nodes, coherence={coherence:.3f}")
        return coherence
    
    def compute_structural_metrics(self) -> Dict[str, float]:
        """Compute structural metrics of the hypergraph."""
        if not self.nodes:
            return self.metrics
        
        # Average degree (hypergraph version)
        total_degree = sum(len(node.incident_edges) for node in self.nodes.values())
        self.metrics['average_degree'] = total_degree / len(self.nodes)
        
        # Attention entropy
        attentions = [node.attention for node in self.nodes.values()]
        total_attention = sum(attentions) + 1e-10
        normalized_attentions = [a / total_attention for a in attentions]
        
        entropy = -sum(p * np.log2(p + 1e-10) for p in normalized_attentions if p > 0)
        self.metrics['attention_entropy'] = entropy
        
        # Pattern count (connected components with high activation)
        active_nodes = [nid for nid, node in self.nodes.items() 
                       if node.activation > self.activation_threshold]
        self.metrics['pattern_count'] = len(self._find_connected_components(active_nodes))
        
        logger.debug(f"Metrics computed: {self.metrics}")
        return self.metrics.copy()
    
    def _find_connected_components(self, node_ids: List[str]) -> List[Set[str]]:
        """Find connected components among given nodes."""
        if not node_ids:
            return []
        
        node_set = set(node_ids)
        visited = set()
        components = []
        
        for node_id in node_ids:
            if node_id in visited:
                continue
            
            # BFS to find component
            component = set()
            queue = deque([node_id])
            
            while queue:
                current = queue.popleft()
                if current in visited:
                    continue
                
                visited.add(current)
                component.add(current)
                
                # Find neighbors through hyperedges
                if current in self.nodes:
                    for edge_id in self.nodes[current].incident_edges:
                        edge = self.edges[edge_id]
                        for neighbor in edge.nodes:
                            if neighbor in node_set and neighbor not in visited:
                                queue.append(neighbor)
            
            if component:
                components.append(component)
        
        return components
    
    def decay_attention(self, decay_rate: Optional[float] = None):
        """Apply attention decay to all nodes."""
        decay = decay_rate if decay_rate is not None else self.attention_decay
        
        for node in self.nodes.values():
            node.attention *= (1.0 - decay)
            if node.attention < 0.01:
                node.attention = 0.0
        
        logger.debug(f"Attention decay applied (rate={decay})")
    
    def get_high_attention_nodes(self, top_k: int = 10) -> List[Tuple[str, float]]:
        """Get nodes with highest attention values."""
        node_attentions = [(nid, node.attention) 
                          for nid, node in self.nodes.items()]
        node_attentions.sort(key=lambda x: x[1], reverse=True)
        return node_attentions[:top_k]
    
    def export_state(self) -> Dict[str, Any]:
        """Export current hypergraph state for persistence or analysis."""
        return {
            'timestamp': datetime.now().isoformat(),
            'metrics': self.metrics,
            'node_count': len(self.nodes),
            'edge_count': len(self.edges),
            'nodes': {
                nid: {
                    'type': node.node_type,
                    'activation': node.activation,
                    'attention': node.attention,
                    'degree': len(node.incident_edges)
                }
                for nid, node in self.nodes.items()
            },
            'edges': {
                eid: {
                    'type': edge.edge_type,
                    'nodes': list(edge.nodes),
                    'weight': edge.weight
                }
                for eid, edge in self.edges.items()
            }
        }
    
    def import_state(self, state: Dict[str, Any]):
        """Import hypergraph state from exported data."""
        # Clear current state
        self.nodes.clear()
        self.edges.clear()
        
        # Import nodes
        for node_id, node_data in state.get('nodes', {}).items():
            node = self.add_node(node_id, node_data['type'])
            node.activation = node_data.get('activation', 0.0)
            node.attention = node_data.get('attention', 0.0)
        
        # Import edges
        for edge_id, edge_data in state.get('edges', {}).items():
            self.add_edge(
                edge_id,
                set(edge_data['nodes']),
                edge_data['type'],
                edge_data.get('weight', 1.0)
            )
        
        logger.info(f"Imported state: {len(self.nodes)} nodes, {len(self.edges)} edges")


class HypergraphSynergyBridge:
    """
    Bridge between hypergraph dynamics and cognitive synergy framework.
    Enables integration with AAR architecture and multi-agent systems.
    """
    
    def __init__(self, hypergraph: HypergraphDynamics):
        self.hypergraph = hypergraph
        self.synergy_metrics: Dict[str, float] = {}
        logger.info("Hypergraph Synergy Bridge initialized")
    
    def integrate_aar_state(self, agent_activations: Dict[str, float],
                           arena_knowledge: Dict[str, Any]) -> Dict[str, float]:
        """
        Integrate AAR (Agent-Arena-Relation) state into hypergraph.
        
        Args:
            agent_activations: Process activations from Agent component
            arena_knowledge: Knowledge state from Arena component
            
        Returns:
            Synergy metrics indicating integration quality
        """
        # Map agent activations to hypergraph nodes
        agent_nodes = []
        for process_id, activation in agent_activations.items():
            node_id = f"agent:{process_id}"
            self.hypergraph.add_node(node_id, "agent_process")
            self.hypergraph.nodes[node_id].activation = activation
            agent_nodes.append(node_id)
        
        # Spread attention from active agent processes
        if agent_nodes:
            attention_map = self.hypergraph.spread_attention(agent_nodes)
            
            # Compute synergy metrics
            self.synergy_metrics['agent_coverage'] = len(attention_map) / max(len(self.hypergraph.nodes), 1)
            self.synergy_metrics['average_attention'] = np.mean(list(attention_map.values()))
        
        return self.synergy_metrics
    
    def extract_patterns_for_synergy(self, min_coherence: float = 0.5) -> List[Dict[str, Any]]:
        """
        Extract high-coherence patterns for cognitive synergy.
        
        Returns list of patterns that can facilitate inter-component collaboration.
        """
        patterns = []
        
        # Find high-attention clusters
        high_attention = self.hypergraph.get_high_attention_nodes(top_k=50)
        
        if len(high_attention) < 2:
            return patterns
        
        # Group into patterns based on connectivity
        node_ids = [nid for nid, _ in high_attention]
        components = self.hypergraph._find_connected_components(node_ids)
        
        for i, component in enumerate(components):
            if len(component) < 2:
                continue
            
            coherence = self.hypergraph.activate_pattern(list(component))
            
            if coherence >= min_coherence:
                patterns.append({
                    'pattern_id': f"pattern_{i}",
                    'nodes': list(component),
                    'size': len(component),
                    'coherence': coherence,
                    'timestamp': datetime.now().isoformat()
                })
        
        logger.info(f"Extracted {len(patterns)} patterns for synergy")
        return patterns


# Example usage and testing
if __name__ == "__main__":
    print("=== Hypergraph Dynamics Module Test ===\n")
    
    # Create hypergraph
    hg = HypergraphDynamics()
    
    # Add nodes
    for i in range(10):
        hg.add_node(f"concept_{i}", "concept")
    
    # Add hyperedges
    hg.add_edge("edge_0", {"concept_0", "concept_1", "concept_2"}, "relation")
    hg.add_edge("edge_1", {"concept_1", "concept_3", "concept_4"}, "relation")
    hg.add_edge("edge_2", {"concept_2", "concept_5"}, "link")
    
    # Spread attention
    attention = hg.spread_attention(["concept_0"], initial_attention=1.0)
    print(f"Attention spread to {len(attention)} nodes")
    
    # Activate pattern
    coherence = hg.activate_pattern(["concept_0", "concept_1", "concept_2"])
    print(f"Pattern coherence: {coherence:.3f}")
    
    # Compute metrics
    metrics = hg.compute_structural_metrics()
    print(f"\nMetrics: {json.dumps(metrics, indent=2)}")
    
    # Test synergy bridge
    bridge = HypergraphSynergyBridge(hg)
    synergy = bridge.integrate_aar_state(
        {"process_a": 0.8, "process_b": 0.6},
        {"knowledge_items": 100}
    )
    print(f"\nSynergy metrics: {json.dumps(synergy, indent=2)}")
    
    print("\n✓ Hypergraph Dynamics module test complete")

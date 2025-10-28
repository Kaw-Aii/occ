"""
Cognitive Synergy Orchestrator for OpenCog Collection
=====================================================

This module provides a unified orchestration layer for cognitive synergy,
coordinating interactions between different cognitive components:
- AAR (Agent-Arena-Relation) self-awareness system
- Hypergraph dynamics and pattern recognition
- Meta-cognitive reasoning
- Neural-symbolic integration
- Multi-agent collaboration

The orchestrator implements the cognitive synergy principle where diverse
AI components interact to produce emergent intelligence beyond individual
capabilities.

Key Responsibilities:
1. Coordinate information flow between cognitive components
2. Detect and resolve bottlenecks in cognitive processes
3. Facilitate pattern sharing and knowledge transfer
4. Monitor and optimize cognitive synergy metrics
5. Provide unified interface for cognitive operations

Author: OpenCog Collection Contributors
License: GPL-3.0+
"""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Callable
from dataclasses import dataclass, field
from datetime import datetime
import logging
import json
from collections import defaultdict, deque
import threading
from queue import Queue, PriorityQueue

# Import cognitive components
try:
    from self_awareness_aar import AARCore
    from hypergraph_dynamics import HypergraphDynamics, HypergraphSynergyBridge
    from cognitive_synergy_framework import HypergraphMemory, CognitiveProcess
except ImportError as e:
    logging.warning(f"Some cognitive modules not available: {e}")
    # Define minimal stubs for testing
    AARCore = None
    HypergraphDynamics = None

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class SynergyMetrics:
    """Metrics tracking cognitive synergy performance."""
    timestamp: datetime = field(default_factory=datetime.now)
    
    # Component health
    aar_coherence: float = 0.0
    hypergraph_connectivity: float = 0.0
    pattern_recognition_rate: float = 0.0
    
    # Synergy indicators
    inter_component_flow: float = 0.0
    bottleneck_count: int = 0
    emergent_pattern_count: int = 0
    
    # Performance
    processing_throughput: float = 0.0
    cognitive_efficiency: float = 0.0
    synergy_index: float = 0.0
    
    def compute_synergy_index(self) -> float:
        """Compute overall synergy index from component metrics."""
        # Weighted combination of metrics
        weights = {
            'aar_coherence': 0.25,
            'hypergraph_connectivity': 0.20,
            'pattern_recognition_rate': 0.15,
            'inter_component_flow': 0.20,
            'cognitive_efficiency': 0.20
        }
        
        self.synergy_index = (
            weights['aar_coherence'] * self.aar_coherence +
            weights['hypergraph_connectivity'] * self.hypergraph_connectivity +
            weights['pattern_recognition_rate'] * self.pattern_recognition_rate +
            weights['inter_component_flow'] * self.inter_component_flow +
            weights['cognitive_efficiency'] * self.cognitive_efficiency
        )
        
        return self.synergy_index


@dataclass
class CognitiveTask:
    """Represents a cognitive task to be processed."""
    task_id: str
    task_type: str
    priority: float
    data: Dict[str, Any]
    required_components: List[str]
    timestamp: datetime = field(default_factory=datetime.now)
    status: str = "pending"
    result: Optional[Any] = None


class CognitiveSynergyOrchestrator:
    """
    Orchestrates cognitive synergy across multiple AI components.
    
    Provides unified coordination for:
    - AAR self-awareness system
    - Hypergraph dynamics
    - Pattern recognition and sharing
    - Meta-cognitive reasoning
    - Multi-agent collaboration
    """
    
    def __init__(self, enable_aar: bool = True, enable_hypergraph: bool = True):
        """
        Initialize cognitive synergy orchestrator.
        
        Args:
            enable_aar: Enable AAR self-awareness component
            enable_hypergraph: Enable hypergraph dynamics component
        """
        # Initialize cognitive components
        self.aar_core = None
        self.hypergraph = None
        self.synergy_bridge = None
        
        if enable_aar and AARCore is not None:
            self.aar_core = AARCore()
            logger.info("AAR Core enabled")
        
        if enable_hypergraph and HypergraphDynamics is not None:
            self.hypergraph = HypergraphDynamics()
            self.synergy_bridge = HypergraphSynergyBridge(self.hypergraph)
            logger.info("Hypergraph Dynamics enabled")
        
        # Task management
        self.task_queue: PriorityQueue = PriorityQueue()
        self.active_tasks: Dict[str, CognitiveTask] = {}
        self.completed_tasks: deque = deque(maxlen=1000)
        
        # Metrics and monitoring
        self.metrics = SynergyMetrics()
        self.metrics_history: deque = deque(maxlen=500)
        
        # Component registry
        self.components: Dict[str, Any] = {
            'aar': self.aar_core,
            'hypergraph': self.hypergraph,
            'synergy_bridge': self.synergy_bridge
        }
        
        # Pattern cache for synergy
        self.shared_patterns: Dict[str, Any] = {}
        self.pattern_usage_count: Dict[str, int] = defaultdict(int)
        
        # Bottleneck detection
        self.bottleneck_threshold = 0.8
        self.bottleneck_history: deque = deque(maxlen=100)
        
        # Threading
        self.lock = threading.RLock()
        self.running = False
        
        logger.info("Cognitive Synergy Orchestrator initialized")
    
    def submit_task(self, task: CognitiveTask) -> str:
        """Submit a cognitive task for processing."""
        with self.lock:
            self.task_queue.put((-task.priority, task.task_id, task))
            self.active_tasks[task.task_id] = task
            logger.info(f"Task submitted: {task.task_id} (priority={task.priority})")
            return task.task_id
    
    def process_task(self, task: CognitiveTask) -> Any:
        """
        Process a cognitive task using appropriate components.
        
        Coordinates between components to achieve cognitive synergy.
        """
        logger.info(f"Processing task: {task.task_id} (type={task.task_type})")
        task.status = "processing"
        
        result = None
        
        try:
            if task.task_type == "pattern_recognition":
                result = self._process_pattern_recognition(task)
            
            elif task.task_type == "self_awareness":
                result = self._process_self_awareness(task)
            
            elif task.task_type == "knowledge_integration":
                result = self._process_knowledge_integration(task)
            
            elif task.task_type == "synergy_optimization":
                result = self._process_synergy_optimization(task)
            
            else:
                logger.warning(f"Unknown task type: {task.task_type}")
                result = {"error": "Unknown task type"}
            
            task.status = "completed"
            task.result = result
            
        except Exception as e:
            logger.error(f"Task processing error: {e}")
            task.status = "failed"
            task.result = {"error": str(e)}
        
        finally:
            with self.lock:
                if task.task_id in self.active_tasks:
                    del self.active_tasks[task.task_id]
                self.completed_tasks.append(task)
        
        return result
    
    def _process_pattern_recognition(self, task: CognitiveTask) -> Dict[str, Any]:
        """Process pattern recognition task using hypergraph dynamics."""
        if not self.hypergraph:
            return {"error": "Hypergraph not available"}
        
        data = task.data
        source_nodes = data.get('source_nodes', [])
        
        # Spread attention to identify patterns
        attention_map = self.hypergraph.spread_attention(source_nodes)
        
        # Extract high-coherence patterns
        if self.synergy_bridge:
            patterns = self.synergy_bridge.extract_patterns_for_synergy()
        else:
            patterns = []
        
        # Cache patterns for reuse
        for pattern in patterns:
            pattern_id = pattern['pattern_id']
            self.shared_patterns[pattern_id] = pattern
            self.pattern_usage_count[pattern_id] += 1
        
        return {
            'attention_spread': len(attention_map),
            'patterns_found': len(patterns),
            'patterns': patterns[:10]  # Return top 10
        }
    
    def _process_self_awareness(self, task: CognitiveTask) -> Dict[str, Any]:
        """Process self-awareness task using AAR core."""
        if not self.aar_core:
            return {"error": "AAR Core not available"}
        
        # Perform meta-cognitive step
        assessment = self.aar_core.meta_cognitive_step()
        
        # Integrate with hypergraph if available
        if self.hypergraph and self.synergy_bridge:
            agent_activations = self.aar_core.agent.process_activations
            arena_knowledge = {
                'density': self.aar_core.arena.knowledge_density,
                'coherence': self.aar_core.arena.coherence_measure
            }
            
            synergy_metrics = self.synergy_bridge.integrate_aar_state(
                agent_activations, arena_knowledge
            )
            assessment['synergy_metrics'] = synergy_metrics
        
        return assessment
    
    def _process_knowledge_integration(self, task: CognitiveTask) -> Dict[str, Any]:
        """Process knowledge integration across components."""
        data = task.data
        knowledge_items = data.get('knowledge_items', [])
        
        integrated_count = 0
        
        # Add to hypergraph
        if self.hypergraph:
            for item in knowledge_items:
                node_id = item.get('id', f"knowledge_{integrated_count}")
                node_type = item.get('type', 'concept')
                self.hypergraph.add_node(node_id, node_type)
                integrated_count += 1
        
        # Update AAR arena
        if self.aar_core:
            self.aar_core.update_arena(
                atom_count=integrated_count,
                link_count=integrated_count // 2,
                pattern_matches=integrated_count,
                total_patterns=len(knowledge_items)
            )
        
        return {
            'integrated_items': integrated_count,
            'total_items': len(knowledge_items),
            'success_rate': integrated_count / max(len(knowledge_items), 1)
        }
    
    def _process_synergy_optimization(self, task: CognitiveTask) -> Dict[str, Any]:
        """Optimize cognitive synergy across components."""
        # Detect bottlenecks
        bottlenecks = self.detect_bottlenecks()
        
        # Apply optimization strategies
        optimizations_applied = []
        
        for bottleneck in bottlenecks:
            if bottleneck['type'] == 'attention_imbalance':
                if self.hypergraph:
                    self.hypergraph.decay_attention(decay_rate=0.2)
                    optimizations_applied.append('attention_decay')
            
            elif bottleneck['type'] == 'low_coherence':
                if self.aar_core:
                    self.aar_core.update_relation()
                    optimizations_applied.append('coherence_update')
        
        # Recompute metrics
        self.update_metrics()
        
        return {
            'bottlenecks_found': len(bottlenecks),
            'optimizations_applied': optimizations_applied,
            'new_synergy_index': self.metrics.synergy_index
        }
    
    def detect_bottlenecks(self) -> List[Dict[str, Any]]:
        """Detect bottlenecks in cognitive processing."""
        bottlenecks = []
        
        # Check AAR coherence
        if self.aar_core:
            if self.aar_core.relation.self_coherence < 0.5:
                bottlenecks.append({
                    'type': 'low_coherence',
                    'component': 'aar',
                    'severity': 1.0 - self.aar_core.relation.self_coherence,
                    'recommendation': 'Update relation state and agent-arena alignment'
                })
        
        # Check hypergraph attention distribution
        if self.hypergraph:
            high_attention = self.hypergraph.get_high_attention_nodes(top_k=10)
            if high_attention:
                top_attention = high_attention[0][1]
                if top_attention > 0.9:
                    bottlenecks.append({
                        'type': 'attention_imbalance',
                        'component': 'hypergraph',
                        'severity': top_attention,
                        'recommendation': 'Apply attention decay to redistribute focus'
                    })
        
        # Check task queue depth
        if self.task_queue.qsize() > 100:
            bottlenecks.append({
                'type': 'task_backlog',
                'component': 'orchestrator',
                'severity': min(self.task_queue.qsize() / 1000, 1.0),
                'recommendation': 'Increase processing capacity or prioritize tasks'
            })
        
        self.bottleneck_history.extend(bottlenecks)
        return bottlenecks
    
    def update_metrics(self):
        """Update cognitive synergy metrics."""
        # AAR metrics
        if self.aar_core:
            self.metrics.aar_coherence = self.aar_core.relation.self_coherence
        
        # Hypergraph metrics
        if self.hypergraph:
            hg_metrics = self.hypergraph.compute_structural_metrics()
            self.metrics.hypergraph_connectivity = min(
                hg_metrics.get('average_degree', 0.0) / 10.0, 1.0
            )
            self.metrics.pattern_recognition_rate = min(
                hg_metrics.get('pattern_count', 0) / 100.0, 1.0
            )
        
        # Inter-component flow
        if self.synergy_bridge:
            synergy = self.synergy_bridge.synergy_metrics
            self.metrics.inter_component_flow = synergy.get('agent_coverage', 0.0)
        
        # Processing metrics
        completed_recent = len([t for t in self.completed_tasks 
                               if (datetime.now() - t.timestamp).seconds < 60])
        self.metrics.processing_throughput = completed_recent / 60.0
        
        # Cognitive efficiency
        if len(self.completed_tasks) > 0:
            success_rate = len([t for t in self.completed_tasks 
                               if t.status == "completed"]) / len(self.completed_tasks)
            self.metrics.cognitive_efficiency = success_rate
        
        # Bottlenecks
        self.metrics.bottleneck_count = len(self.detect_bottlenecks())
        
        # Compute synergy index
        self.metrics.compute_synergy_index()
        
        # Record metrics
        self.metrics_history.append(self.metrics)
        
        logger.debug(f"Metrics updated: synergy_index={self.metrics.synergy_index:.3f}")
    
    def get_status(self) -> Dict[str, Any]:
        """Get comprehensive orchestrator status."""
        return {
            'timestamp': datetime.now().isoformat(),
            'components': {
                'aar_enabled': self.aar_core is not None,
                'hypergraph_enabled': self.hypergraph is not None,
            },
            'tasks': {
                'queued': self.task_queue.qsize(),
                'active': len(self.active_tasks),
                'completed': len(self.completed_tasks)
            },
            'metrics': {
                'synergy_index': self.metrics.synergy_index,
                'aar_coherence': self.metrics.aar_coherence,
                'hypergraph_connectivity': self.metrics.hypergraph_connectivity,
                'bottleneck_count': self.metrics.bottleneck_count,
                'cognitive_efficiency': self.metrics.cognitive_efficiency
            },
            'patterns': {
                'shared_patterns': len(self.shared_patterns),
                'total_usage': sum(self.pattern_usage_count.values())
            }
        }
    
    def export_state(self) -> Dict[str, Any]:
        """Export complete orchestrator state."""
        state = {
            'timestamp': datetime.now().isoformat(),
            'status': self.get_status(),
            'metrics_history': [
                {
                    'timestamp': m.timestamp.isoformat(),
                    'synergy_index': m.synergy_index,
                    'aar_coherence': m.aar_coherence,
                    'hypergraph_connectivity': m.hypergraph_connectivity
                }
                for m in list(self.metrics_history)[-50:]
            ]
        }
        
        # Export component states
        if self.hypergraph:
            state['hypergraph_state'] = self.hypergraph.export_state()
        
        if self.aar_core:
            state['aar_state'] = self.aar_core.perceive_self()
        
        return state


# Example usage and testing
if __name__ == "__main__":
    print("=== Cognitive Synergy Orchestrator Test ===\n")
    
    # Create orchestrator
    orchestrator = CognitiveSynergyOrchestrator(
        enable_aar=True,
        enable_hypergraph=True
    )
    
    # Submit test tasks
    task1 = CognitiveTask(
        task_id="task_1",
        task_type="pattern_recognition",
        priority=0.8,
        data={'source_nodes': ['concept_0', 'concept_1']},
        required_components=['hypergraph']
    )
    
    task2 = CognitiveTask(
        task_id="task_2",
        task_type="self_awareness",
        priority=0.9,
        data={},
        required_components=['aar']
    )
    
    orchestrator.submit_task(task1)
    orchestrator.submit_task(task2)
    
    # Process tasks
    print("Processing tasks...")
    while not orchestrator.task_queue.empty():
        _, _, task = orchestrator.task_queue.get()
        result = orchestrator.process_task(task)
        print(f"Task {task.task_id} completed: {task.status}")
    
    # Update metrics
    orchestrator.update_metrics()
    
    # Get status
    status = orchestrator.get_status()
    print(f"\nOrchestrator Status:")
    print(json.dumps(status, indent=2))
    
    print("\n✓ Cognitive Synergy Orchestrator test complete")

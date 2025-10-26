"""
Cognitive Orchestrator - Unified Coordination System
====================================================

This module implements the central orchestration system that coordinates all
cognitive components in the OpenCog Collection through attention-based resource
allocation and synergy monitoring.

The orchestrator integrates:
- AAR Core (Self-awareness)
- Membrane System (Process organization)
- Attention Broker (Resource allocation)
- Synergy Monitor (Emergent capability tracking)
- Feedback Router (Cross-component learning)

This creates a unified cognitive architecture where components collaborate
through the hypergraph memory substrate, enabling true cognitive synergy.

Author: OpenCog Collection Contributors
License: GPL-3.0+
"""

import logging
from typing import Dict, List, Any, Optional, Set, Callable, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from collections import defaultdict, deque
import numpy as np
import threading
from queue import Queue, PriorityQueue
import json

# Import existing OCC modules
try:
    from self_awareness_aar import AARCore, AgentState, ArenaState, RelationState
except ImportError:
    logging.warning("AAR module not found, using stub")
    AARCore = None

try:
    from deep_tree_echo_membranes import Membrane, MembraneMessage, MembraneType
except ImportError:
    logging.warning("Membrane module not found, using stub")
    Membrane = None

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ComponentDescriptor:
    """Describes a cognitive component in the system."""
    component_id: str
    component_type: str  # 'symbolic', 'neural', 'evolutionary', 'sensory'
    capabilities: Set[str]
    resource_requirements: Dict[str, float]
    current_activation: float = 0.0
    performance_history: deque = field(default_factory=lambda: deque(maxlen=100))
    last_active: datetime = field(default_factory=datetime.now)


@dataclass
class AttentionAllocation:
    """Represents attention allocation to a component."""
    component_id: str
    attention_weight: float
    reason: str
    timestamp: datetime = field(default_factory=datetime.now)
    expected_duration: float = 1.0  # seconds


@dataclass
class SynergyEvent:
    """Records an instance of cognitive synergy."""
    event_id: str
    components_involved: List[str]
    synergy_type: str  # 'bottleneck_resolution', 'pattern_sharing', 'cross_validation'
    effectiveness_score: float
    description: str
    timestamp: datetime = field(default_factory=datetime.now)


class AttentionBroker:
    """
    Manages attention allocation across cognitive components.
    
    Implements dynamic resource allocation based on:
    - Component urgency and importance
    - Bottleneck detection
    - Synergy opportunities
    - Historical performance
    """
    
    def __init__(self, total_attention: float = 100.0):
        self.total_attention = total_attention
        self.allocations: Dict[str, AttentionAllocation] = {}
        self.allocation_history: deque = deque(maxlen=1000)
        self.urgency_scores: Dict[str, float] = {}
        
        logger.info(f"AttentionBroker initialized with {total_attention} units")
    
    def compute_urgency(self, component: ComponentDescriptor,
                       bottleneck_detected: bool = False,
                       synergy_opportunity: bool = False) -> float:
        """Compute urgency score for a component."""
        base_urgency = component.current_activation
        
        # Boost for bottlenecks
        if bottleneck_detected:
            base_urgency *= 2.0
        
        # Boost for synergy opportunities
        if synergy_opportunity:
            base_urgency *= 1.5
        
        # Decay based on time since last active
        time_since_active = (datetime.now() - component.last_active).total_seconds()
        decay_factor = np.exp(-time_since_active / 60.0)  # 1-minute half-life
        
        return base_urgency * decay_factor
    
    def allocate_attention(self, components: Dict[str, ComponentDescriptor],
                          bottlenecks: Set[str] = None,
                          synergy_opportunities: Set[str] = None) -> Dict[str, float]:
        """
        Allocate attention across components based on urgency and opportunity.
        
        Returns: Dict mapping component_id to attention weight
        """
        if bottlenecks is None:
            bottlenecks = set()
        if synergy_opportunities is None:
            synergy_opportunities = set()
        
        # Compute urgency scores
        urgency_scores = {}
        for comp_id, component in components.items():
            urgency = self.compute_urgency(
                component,
                bottleneck_detected=(comp_id in bottlenecks),
                synergy_opportunity=(comp_id in synergy_opportunities)
            )
            urgency_scores[comp_id] = urgency
        
        # Normalize to total attention
        total_urgency = sum(urgency_scores.values())
        if total_urgency == 0:
            return {comp_id: 0.0 for comp_id in components}
        
        allocations = {
            comp_id: (urgency / total_urgency) * self.total_attention
            for comp_id, urgency in urgency_scores.items()
        }
        
        # Record allocations
        for comp_id, attention in allocations.items():
            allocation = AttentionAllocation(
                component_id=comp_id,
                attention_weight=attention,
                reason=self._determine_allocation_reason(
                    comp_id, bottlenecks, synergy_opportunities
                )
            )
            self.allocations[comp_id] = allocation
            self.allocation_history.append(allocation)
        
        logger.debug(f"Attention allocated across {len(allocations)} components")
        return allocations
    
    def _determine_allocation_reason(self, comp_id: str,
                                     bottlenecks: Set[str],
                                     synergy_opportunities: Set[str]) -> str:
        """Determine why attention was allocated to a component."""
        reasons = []
        if comp_id in bottlenecks:
            reasons.append("bottleneck_resolution")
        if comp_id in synergy_opportunities:
            reasons.append("synergy_opportunity")
        if not reasons:
            reasons.append("normal_activation")
        return ", ".join(reasons)


class SynergyMonitor:
    """
    Monitors and measures cognitive synergy across components.
    
    Tracks:
    - Cross-component interactions
    - Emergent capabilities
    - Synergy effectiveness
    - Pattern sharing events
    """
    
    def __init__(self):
        self.synergy_events: List[SynergyEvent] = []
        self.interaction_graph: Dict[str, Set[str]] = defaultdict(set)
        self.emergent_capabilities: Set[str] = set()
        self.synergy_metrics: Dict[str, float] = {
            'total_synergy_score': 0.0,
            'interaction_density': 0.0,
            'emergent_capability_count': 0.0
        }
        
        logger.info("SynergyMonitor initialized")
    
    def record_interaction(self, component_a: str, component_b: str,
                          interaction_type: str, effectiveness: float):
        """Record an interaction between two components."""
        self.interaction_graph[component_a].add(component_b)
        self.interaction_graph[component_b].add(component_a)
        
        # Create synergy event
        event = SynergyEvent(
            event_id=f"syn_{len(self.synergy_events)}",
            components_involved=[component_a, component_b],
            synergy_type=interaction_type,
            effectiveness_score=effectiveness,
            description=f"{interaction_type} between {component_a} and {component_b}"
        )
        self.synergy_events.append(event)
        
        # Update metrics
        self._update_metrics()
        
        logger.debug(f"Synergy event recorded: {interaction_type} "
                    f"(effectiveness={effectiveness:.3f})")
    
    def detect_emergent_capability(self, capability_name: str,
                                   contributing_components: List[str]):
        """Detect and record an emergent capability."""
        if capability_name not in self.emergent_capabilities:
            self.emergent_capabilities.add(capability_name)
            
            event = SynergyEvent(
                event_id=f"emerg_{len(self.synergy_events)}",
                components_involved=contributing_components,
                synergy_type="emergent_capability",
                effectiveness_score=1.0,
                description=f"Emergent capability: {capability_name}"
            )
            self.synergy_events.append(event)
            
            logger.info(f"Emergent capability detected: {capability_name}")
            self._update_metrics()
    
    def compute_synergy_score(self) -> float:
        """Compute overall synergy effectiveness score."""
        if not self.synergy_events:
            return 0.0
        
        recent_events = list(self.synergy_events)[-100:]
        avg_effectiveness = np.mean([e.effectiveness_score for e in recent_events])
        
        # Factor in interaction density
        total_components = len(self.interaction_graph)
        if total_components > 1:
            max_interactions = total_components * (total_components - 1) / 2
            actual_interactions = sum(len(neighbors) for neighbors in self.interaction_graph.values()) / 2
            interaction_density = actual_interactions / max_interactions
        else:
            interaction_density = 0.0
        
        # Combined score
        synergy_score = 0.7 * avg_effectiveness + 0.3 * interaction_density
        return synergy_score
    
    def _update_metrics(self):
        """Update synergy metrics."""
        self.synergy_metrics['total_synergy_score'] = self.compute_synergy_score()
        
        total_components = len(self.interaction_graph)
        if total_components > 1:
            max_interactions = total_components * (total_components - 1) / 2
            actual_interactions = sum(len(neighbors) for neighbors in self.interaction_graph.values()) / 2
            self.synergy_metrics['interaction_density'] = actual_interactions / max_interactions
        else:
            self.synergy_metrics['interaction_density'] = 0.0
        
        self.synergy_metrics['emergent_capability_count'] = len(self.emergent_capabilities)


class FeedbackRouter:
    """
    Routes feedback and learning signals between components.
    
    Enables cross-component learning through:
    - Pattern sharing
    - Success/failure propagation
    - Cross-validation results
    - Meta-learning signals
    """
    
    def __init__(self):
        self.feedback_queue: Queue = Queue()
        self.learning_signals: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        self.pattern_cache: Dict[str, Any] = {}
        
        logger.info("FeedbackRouter initialized")
    
    def route_pattern(self, source_component: str, target_components: List[str],
                     pattern: Dict[str, Any]):
        """Route a discovered pattern to target components."""
        pattern_id = f"pat_{len(self.pattern_cache)}"
        self.pattern_cache[pattern_id] = {
            'source': source_component,
            'pattern': pattern,
            'timestamp': datetime.now()
        }
        
        for target in target_components:
            signal = {
                'type': 'pattern_sharing',
                'pattern_id': pattern_id,
                'pattern': pattern,
                'source': source_component
            }
            self.learning_signals[target].append(signal)
        
        logger.debug(f"Pattern routed from {source_component} to {len(target_components)} components")
    
    def route_validation_result(self, validator_component: str,
                               validated_component: str,
                               result: Dict[str, Any]):
        """Route cross-validation results."""
        signal = {
            'type': 'cross_validation',
            'validator': validator_component,
            'result': result,
            'timestamp': datetime.now()
        }
        self.learning_signals[validated_component].append(signal)
        
        logger.debug(f"Validation result routed from {validator_component} to {validated_component}")
    
    def get_signals_for_component(self, component_id: str) -> List[Dict[str, Any]]:
        """Retrieve pending learning signals for a component."""
        signals = self.learning_signals.get(component_id, [])
        self.learning_signals[component_id] = []  # Clear after retrieval
        return signals


class CognitiveOrchestrator:
    """
    Central orchestration system for the OpenCog Collection.
    
    Coordinates all cognitive components through:
    - Unified component registry
    - Attention-based resource allocation
    - Synergy monitoring and optimization
    - Cross-component feedback routing
    - Meta-cognitive control via AAR core
    """
    
    def __init__(self):
        # Core subsystems
        self.aar_core = AARCore() if AARCore else None
        self.attention_broker = AttentionBroker()
        self.synergy_monitor = SynergyMonitor()
        self.feedback_router = FeedbackRouter()
        
        # Component registry
        self.components: Dict[str, ComponentDescriptor] = {}
        self.component_handlers: Dict[str, Callable] = {}
        
        # State tracking
        self.orchestration_step = 0
        self.bottlenecks_detected: Set[str] = set()
        self.synergy_opportunities: Set[str] = set()
        
        # Hypergraph memory (placeholder for AtomSpace integration)
        self.hypergraph_memory: Dict[str, Any] = {}
        
        logger.info("CognitiveOrchestrator initialized - unified cognitive architecture ready")
    
    def register_component(self, component_id: str, component_type: str,
                          capabilities: Set[str],
                          handler: Optional[Callable] = None):
        """Register a cognitive component with the orchestrator."""
        descriptor = ComponentDescriptor(
            component_id=component_id,
            component_type=component_type,
            capabilities=capabilities,
            resource_requirements={'attention': 10.0, 'memory': 100.0}
        )
        self.components[component_id] = descriptor
        
        if handler:
            self.component_handlers[component_id] = handler
        
        logger.info(f"Component registered: {component_id} ({component_type})")
    
    def orchestration_cycle(self) -> Dict[str, Any]:
        """
        Execute one orchestration cycle.
        
        Steps:
        1. Update AAR self-awareness
        2. Detect bottlenecks and opportunities
        3. Allocate attention
        4. Route feedback
        5. Monitor synergy
        6. Update components
        
        Returns: Orchestration cycle report
        """
        self.orchestration_step += 1
        cycle_start = datetime.now()
        
        logger.info(f"=== Orchestration Cycle {self.orchestration_step} ===")
        
        # Step 1: Update AAR self-awareness
        if self.aar_core:
            self._update_self_awareness()
        
        # Step 2: Detect bottlenecks and opportunities
        self._detect_bottlenecks()
        self._detect_synergy_opportunities()
        
        # Step 3: Allocate attention
        attention_allocations = self.attention_broker.allocate_attention(
            self.components,
            self.bottlenecks_detected,
            self.synergy_opportunities
        )
        
        # Step 4: Execute components based on attention
        component_results = self._execute_components(attention_allocations)
        
        # Step 5: Route feedback
        self._route_feedback(component_results)
        
        # Step 6: Monitor synergy
        synergy_score = self.synergy_monitor.compute_synergy_score()
        
        # Generate cycle report
        cycle_duration = (datetime.now() - cycle_start).total_seconds()
        report = {
            'cycle': self.orchestration_step,
            'duration': cycle_duration,
            'components_active': len([c for c in attention_allocations.values() if c > 0]),
            'bottlenecks_detected': len(self.bottlenecks_detected),
            'synergy_opportunities': len(self.synergy_opportunities),
            'synergy_score': synergy_score,
            'attention_allocations': attention_allocations,
            'component_results': component_results
        }
        
        logger.info(f"Cycle {self.orchestration_step} complete: "
                   f"synergy_score={synergy_score:.3f}, duration={cycle_duration:.3f}s")
        
        return report
    
    def _update_self_awareness(self):
        """Update AAR self-awareness state."""
        if not self.aar_core:
            return
        
        # Update Agent state with component activations
        process_activations = {
            comp_id: comp.current_activation
            for comp_id, comp in self.components.items()
        }
        self.aar_core.update_agent(process_activations)
        
        # Update Arena state with hypergraph statistics
        atom_count = len(self.hypergraph_memory)
        link_count = sum(1 for v in self.hypergraph_memory.values() 
                        if isinstance(v, (list, dict)))
        self.aar_core.update_arena(atom_count, link_count, 0, 1)
        
        # Update Relation state
        self.aar_core.update_relation()
    
    def _detect_bottlenecks(self):
        """Detect components experiencing bottlenecks."""
        self.bottlenecks_detected.clear()
        
        for comp_id, component in self.components.items():
            # Simple heuristic: high activation but low recent performance
            if component.current_activation > 0.7:
                if component.performance_history:
                    recent_perf = np.mean(list(component.performance_history)[-10:])
                    if recent_perf < 0.5:
                        self.bottlenecks_detected.add(comp_id)
                        logger.info(f"Bottleneck detected in {comp_id}")
    
    def _detect_synergy_opportunities(self):
        """Detect opportunities for cognitive synergy."""
        self.synergy_opportunities.clear()
        
        # Look for complementary capabilities
        for comp_id_a, comp_a in self.components.items():
            for comp_id_b, comp_b in self.components.items():
                if comp_id_a >= comp_id_b:
                    continue
                
                # Check for complementary capabilities
                if comp_a.capabilities & comp_b.capabilities:
                    continue  # Skip if capabilities overlap too much
                
                # Check if both are moderately active
                if 0.3 < comp_a.current_activation < 0.8 and \
                   0.3 < comp_b.current_activation < 0.8:
                    self.synergy_opportunities.add(comp_id_a)
                    self.synergy_opportunities.add(comp_id_b)
    
    def _execute_components(self, attention_allocations: Dict[str, float]) -> Dict[str, Any]:
        """Execute components based on attention allocations."""
        results = {}
        
        for comp_id, attention in attention_allocations.items():
            if attention < 1.0:  # Skip components with minimal attention
                continue
            
            component = self.components[comp_id]
            
            # Execute component handler if available
            if comp_id in self.component_handlers:
                try:
                    result = self.component_handlers[comp_id](attention)
                    results[comp_id] = result
                    
                    # Update performance history
                    performance = result.get('performance', 0.5)
                    component.performance_history.append(performance)
                    
                except Exception as e:
                    logger.error(f"Component {comp_id} execution failed: {e}")
                    results[comp_id] = {'error': str(e), 'performance': 0.0}
            else:
                # Placeholder result
                results[comp_id] = {'status': 'no_handler', 'performance': 0.5}
            
            component.last_active = datetime.now()
        
        return results
    
    def _route_feedback(self, component_results: Dict[str, Any]):
        """Route feedback between components based on results."""
        for comp_id, result in component_results.items():
            if 'pattern' in result:
                # Share discovered patterns
                target_components = [
                    other_id for other_id in self.components
                    if other_id != comp_id
                ]
                self.feedback_router.route_pattern(
                    comp_id, target_components, result['pattern']
                )
                
                # Record synergy event
                for target in target_components:
                    self.synergy_monitor.record_interaction(
                        comp_id, target, 'pattern_sharing', 0.7
                    )
    
    def get_system_state(self) -> Dict[str, Any]:
        """Get comprehensive system state report."""
        state = {
            'orchestration_step': self.orchestration_step,
            'components': {
                comp_id: {
                    'type': comp.component_type,
                    'activation': comp.current_activation,
                    'capabilities': list(comp.capabilities)
                }
                for comp_id, comp in self.components.items()
            },
            'synergy_metrics': self.synergy_monitor.synergy_metrics,
            'bottlenecks': list(self.bottlenecks_detected),
            'synergy_opportunities': list(self.synergy_opportunities)
        }
        
        if self.aar_core:
            state['self_awareness'] = self.aar_core.perceive_self()
        
        return state


# Example usage and demonstration
if __name__ == "__main__":
    print("Cognitive Orchestrator - Unified Cognitive Architecture")
    print("=" * 60)
    
    # Initialize orchestrator
    orchestrator = CognitiveOrchestrator()
    
    # Register example components
    orchestrator.register_component(
        "symbolic_reasoner",
        "symbolic",
        {"reasoning", "logic", "inference"}
    )
    
    orchestrator.register_component(
        "neural_learner",
        "neural",
        {"learning", "pattern_recognition", "prediction"}
    )
    
    orchestrator.register_component(
        "pattern_miner",
        "evolutionary",
        {"pattern_mining", "optimization", "search"}
    )
    
    # Set some initial activations
    orchestrator.components["symbolic_reasoner"].current_activation = 0.6
    orchestrator.components["neural_learner"].current_activation = 0.8
    orchestrator.components["pattern_miner"].current_activation = 0.4
    
    # Run orchestration cycles
    print("\nRunning orchestration cycles...\n")
    for i in range(3):
        report = orchestrator.orchestration_cycle()
        print(f"\nCycle {report['cycle']}:")
        print(f"  Active components: {report['components_active']}")
        print(f"  Synergy score: {report['synergy_score']:.3f}")
        print(f"  Bottlenecks: {report['bottlenecks_detected']}")
        print(f"  Synergy opportunities: {report['synergy_opportunities']}")
    
    # Get final system state
    print("\n" + "=" * 60)
    print("Final System State:")
    state = orchestrator.get_system_state()
    print(json.dumps(state, indent=2, default=str))


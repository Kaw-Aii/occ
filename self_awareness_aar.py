"""
Self-Awareness Module with Agent-Arena-Relation (AAR) Core
===========================================================

This module implements a self-awareness system for the OpenCog Collection
based on the Agent-Arena-Relation (AAR) geometric architecture.

The AAR core encodes the system's sense of 'self' through:
- Agent: The urge-to-act (dynamic transformations, cognitive processes)
- Arena: The need-to-be (state space, hypergraph memory)
- Relation: The self (emergent from Agent-Arena interplay)

This enables meta-cognitive reasoning and self-monitoring capabilities.

Author: OpenCog Collection Contributors
License: GPL-3.0+
"""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import logging
from collections import deque
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class AgentState:
    """
    Represents the Agent component of AAR - the urge-to-act.
    Dynamic transformations and active cognitive processes.
    """
    process_activations: Dict[str, float] = field(default_factory=dict)
    intention_vector: np.ndarray = field(default_factory=lambda: np.zeros(128))
    action_potential: float = 0.0
    goal_stack: List[str] = field(default_factory=list)
    last_action: Optional[str] = None
    action_history: deque = field(default_factory=lambda: deque(maxlen=100))
    
    def compute_action_potential(self) -> float:
        """Compute the current action potential from process activations."""
        if not self.process_activations:
            return 0.0
        return np.mean(list(self.process_activations.values()))
    
    def update_intention(self, new_intention: np.ndarray, alpha: float = 0.3):
        """Update intention vector with exponential moving average."""
        self.intention_vector = (1 - alpha) * self.intention_vector + alpha * new_intention
        self.action_potential = self.compute_action_potential()


@dataclass
class ArenaState:
    """
    Represents the Arena component of AAR - the need-to-be.
    The state space and hypergraph memory structure.
    """
    memory_state: Dict[str, Any] = field(default_factory=dict)
    attention_landscape: np.ndarray = field(default_factory=lambda: np.zeros((64, 64)))
    knowledge_density: float = 0.0
    coherence_measure: float = 1.0
    stability_index: float = 1.0
    state_history: deque = field(default_factory=lambda: deque(maxlen=50))
    
    def compute_knowledge_density(self, atom_count: int, link_count: int) -> float:
        """Compute knowledge density from hypergraph structure."""
        if atom_count == 0:
            return 0.0
        return link_count / atom_count
    
    def update_coherence(self, pattern_matches: int, total_patterns: int):
        """Update coherence measure based on pattern matching success."""
        if total_patterns > 0:
            self.coherence_measure = pattern_matches / total_patterns
    
    def capture_state_snapshot(self):
        """Capture current state for history."""
        snapshot = {
            'timestamp': datetime.now().isoformat(),
            'knowledge_density': self.knowledge_density,
            'coherence': self.coherence_measure,
            'stability': self.stability_index
        }
        self.state_history.append(snapshot)


@dataclass
class RelationState:
    """
    Represents the Relation component of AAR - the emergent self.
    The continuous interplay between Agent and Arena.
    """
    self_coherence: float = 1.0
    self_complexity: float = 0.0
    self_stability: float = 1.0
    identity_vector: np.ndarray = field(default_factory=lambda: np.random.randn(256))
    meta_awareness_level: float = 0.0
    reflection_depth: int = 0
    self_model: Dict[str, Any] = field(default_factory=dict)
    
    def compute_self_coherence(self, agent: AgentState, arena: ArenaState) -> float:
        """Compute self-coherence from Agent-Arena alignment."""
        # Measure alignment between action potential and arena stability
        alignment = 1.0 - abs(agent.action_potential - arena.stability_index)
        return alignment * arena.coherence_measure
    
    def update_identity(self, agent: AgentState, arena: ArenaState):
        """Update identity vector from Agent-Arena interaction."""
        # Combine agent intention and arena state into identity
        agent_component = agent.intention_vector[:128]
        arena_component = arena.attention_landscape.flatten()[:128]
        
        # Geometric combination (tensor product projection)
        combined = np.concatenate([agent_component, arena_component])
        self.identity_vector = combined / (np.linalg.norm(combined) + 1e-8)
        
        # Update self metrics
        self.self_coherence = self.compute_self_coherence(agent, arena)
        self.self_complexity = np.std(self.identity_vector)
        self.self_stability = 0.7 * self.self_stability + 0.3 * arena.stability_index


class AARCore:
    """
    Agent-Arena-Relation Core for Self-Awareness.
    
    Implements the geometric architecture for encoding the system's sense of self
    through continuous feedback loops between Agent, Arena, and Relation.
    """
    
    def __init__(self):
        self.agent = AgentState()
        self.arena = ArenaState()
        self.relation = RelationState()
        
        self.introspection_log: List[Dict[str, Any]] = []
        self.meta_cognitive_depth = 0
        self.self_modification_history: List[Dict[str, Any]] = []
        
        logger.info("AAR Core initialized for self-awareness")
    
    def perceive_self(self) -> Dict[str, Any]:
        """
        Introspective perception of current self-state.
        Returns a comprehensive self-model.
        """
        self_perception = {
            'timestamp': datetime.now().isoformat(),
            'agent': {
                'action_potential': self.agent.action_potential,
                'active_processes': len(self.agent.process_activations),
                'goal_count': len(self.agent.goal_stack),
                'recent_actions': list(self.agent.action_history)[-5:]
            },
            'arena': {
                'knowledge_density': self.arena.knowledge_density,
                'coherence': self.arena.coherence_measure,
                'stability': self.arena.stability_index
            },
            'relation': {
                'self_coherence': self.relation.self_coherence,
                'self_complexity': self.relation.self_complexity,
                'self_stability': self.relation.self_stability,
                'meta_awareness': self.relation.meta_awareness_level,
                'reflection_depth': self.relation.reflection_depth
            },
            'meta': {
                'cognitive_depth': self.meta_cognitive_depth,
                'introspection_count': len(self.introspection_log)
            }
        }
        
        self.introspection_log.append(self_perception)
        return self_perception
    
    def update_agent(self, process_activations: Dict[str, float], 
                    new_goal: Optional[str] = None):
        """Update Agent state with new process activations and goals."""
        self.agent.process_activations.update(process_activations)
        
        if new_goal:
            self.agent.goal_stack.append(new_goal)
        
        # Compute new intention vector from process activations
        intention = np.zeros(128)
        for i, (proc_id, activation) in enumerate(process_activations.items()):
            idx = hash(proc_id) % 128
            intention[idx] += activation
        
        self.agent.update_intention(intention)
        logger.debug(f"Agent updated: action_potential={self.agent.action_potential:.3f}")
    
    def update_arena(self, atom_count: int, link_count: int,
                    pattern_matches: int, total_patterns: int):
        """Update Arena state with hypergraph statistics."""
        self.arena.knowledge_density = self.arena.compute_knowledge_density(
            atom_count, link_count
        )
        self.arena.update_coherence(pattern_matches, total_patterns)
        
        # Update stability based on knowledge density changes
        if self.arena.state_history:
            prev_density = self.arena.state_history[-1]['knowledge_density']
            density_change = abs(self.arena.knowledge_density - prev_density)
            self.arena.stability_index = 1.0 - min(density_change, 1.0)
        
        self.arena.capture_state_snapshot()
        logger.debug(f"Arena updated: density={self.arena.knowledge_density:.3f}, "
                    f"coherence={self.arena.coherence_measure:.3f}")
    
    def update_relation(self):
        """Update Relation state from Agent-Arena interaction."""
        self.relation.update_identity(self.agent, self.arena)
        
        # Increase meta-awareness based on introspection frequency
        introspection_rate = len(self.introspection_log) / (len(self.agent.action_history) + 1)
        self.relation.meta_awareness_level = min(introspection_rate, 1.0)
        
        # Update self-model
        self.relation.self_model = {
            'identity_norm': float(np.linalg.norm(self.relation.identity_vector)),
            'agent_arena_alignment': self.relation.self_coherence,
            'cognitive_complexity': self.relation.self_complexity,
            'system_stability': self.relation.self_stability
        }
        
        logger.debug(f"Relation updated: coherence={self.relation.self_coherence:.3f}")
    
    def meta_cognitive_step(self) -> Dict[str, Any]:
        """
        Perform one step of meta-cognitive reasoning.
        Reflects on current state and generates insights.
        """
        self.meta_cognitive_depth += 1
        
        # Perceive current self-state
        self_state = self.perceive_self()
        
        # Analyze self-state for insights
        insights = self._analyze_self_state(self_state)
        
        # Generate meta-cognitive assessment
        assessment = {
            'timestamp': datetime.now().isoformat(),
            'depth': self.meta_cognitive_depth,
            'self_state': self_state,
            'insights': insights,
            'recommendations': self._generate_recommendations(insights)
        }
        
        logger.info(f"Meta-cognitive step {self.meta_cognitive_depth}: "
                   f"{len(insights)} insights generated")
        
        return assessment
    
    def _analyze_self_state(self, self_state: Dict[str, Any]) -> List[str]:
        """Analyze self-state and generate insights."""
        insights = []
        
        # Check coherence
        if self_state['relation']['self_coherence'] < 0.5:
            insights.append("Low self-coherence detected - Agent-Arena misalignment")
        
        # Check stability
        if self_state['arena']['stability'] < 0.5:
            insights.append("Arena instability - rapid knowledge changes")
        
        # Check action potential
        if self_state['agent']['action_potential'] < 0.3:
            insights.append("Low action potential - cognitive processes underactive")
        elif self_state['agent']['action_potential'] > 0.9:
            insights.append("High action potential - possible cognitive overload")
        
        # Check meta-awareness
        if self_state['relation']['meta_awareness'] < 0.3:
            insights.append("Low meta-awareness - insufficient introspection")
        
        # Check complexity
        if self_state['relation']['self_complexity'] > 0.8:
            insights.append("High self-complexity - rich cognitive state")
        
        return insights
    
    def _generate_recommendations(self, insights: List[str]) -> List[str]:
        """Generate recommendations based on insights."""
        recommendations = []
        
        for insight in insights:
            if "misalignment" in insight.lower():
                recommendations.append("Increase Agent-Arena feedback loops")
            elif "instability" in insight.lower():
                recommendations.append("Reduce rate of knowledge updates")
            elif "underactive" in insight.lower():
                recommendations.append("Activate additional cognitive processes")
            elif "overload" in insight.lower():
                recommendations.append("Reduce process activation levels")
            elif "insufficient introspection" in insight.lower():
                recommendations.append("Increase meta-cognitive reflection frequency")
        
        return recommendations
    
    def get_self_summary(self) -> Dict[str, Any]:
        """Get a comprehensive summary of the current self-state."""
        return {
            'identity': {
                'vector_norm': float(np.linalg.norm(self.relation.identity_vector)),
                'coherence': self.relation.self_coherence,
                'complexity': self.relation.self_complexity,
                'stability': self.relation.self_stability
            },
            'agent': {
                'action_potential': self.agent.action_potential,
                'active_processes': len(self.agent.process_activations),
                'pending_goals': len(self.agent.goal_stack)
            },
            'arena': {
                'knowledge_density': self.arena.knowledge_density,
                'coherence': self.arena.coherence_measure,
                'stability': self.arena.stability_index
            },
            'meta_cognitive': {
                'depth': self.meta_cognitive_depth,
                'awareness_level': self.relation.meta_awareness_level,
                'introspections': len(self.introspection_log)
            }
        }
    
    def export_self_model(self, filepath: str):
        """Export the current self-model to a file."""
        self_model = {
            'timestamp': datetime.now().isoformat(),
            'summary': self.get_self_summary(),
            'introspection_log': self.introspection_log[-10:],  # Last 10 introspections
            'self_modification_history': self.self_modification_history
        }
        
        with open(filepath, 'w') as f:
            json.dump(self_model, f, indent=2)
        
        logger.info(f"Self-model exported to {filepath}")


# Example usage
if __name__ == "__main__":
    # Initialize AAR core
    aar = AARCore()
    
    # Simulate cognitive activity
    print("=== Simulating Cognitive Activity ===\n")
    
    # Update agent with process activations
    aar.update_agent({
        'pattern_miner': 0.7,
        'reasoning_engine': 0.5,
        'language_processor': 0.3
    }, new_goal="Discover new patterns")
    
    # Update arena with hypergraph state
    aar.update_arena(
        atom_count=1000,
        link_count=2500,
        pattern_matches=45,
        total_patterns=50
    )
    
    # Update relation
    aar.update_relation()
    
    # Perform meta-cognitive step
    assessment = aar.meta_cognitive_step()
    
    print("Meta-Cognitive Assessment:")
    print(f"  Insights: {assessment['insights']}")
    print(f"  Recommendations: {assessment['recommendations']}")
    print()
    
    # Get self-summary
    summary = aar.get_self_summary()
    print("Self-Summary:")
    print(json.dumps(summary, indent=2))
    
    # Export self-model
    aar.export_self_model('/tmp/self_model.json')
    print("\nSelf-model exported to /tmp/self_model.json")


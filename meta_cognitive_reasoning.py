#!/usr/bin/env python3
"""
Meta-Cognitive Reasoning Module for OpenCog Collection
=====================================================

This module implements meta-cognitive reasoning capabilities that enable
the system to reason about its own reasoning processes, monitor cognitive
performance, and adapt cognitive strategies dynamically.

Key Features:
- Self-monitoring of cognitive processes
- Strategy adaptation based on performance
- Meta-level pattern recognition
- Cognitive resource allocation optimization
- Self-reflective learning mechanisms

Author: OpenCog Collection Contributors
License: GPL-3.0+
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import json
import logging
from datetime import datetime, timedelta
import threading
from queue import Queue, PriorityQueue
from collections import defaultdict, deque
from enum import Enum
import statistics

from cognitive_synergy_framework import (
    CognitiveSynergyEngine, CognitiveProcess, Atom, 
    HypergraphMemory, PatternMiner
)

logger = logging.getLogger(__name__)


class CognitiveState(Enum):
    """Enumeration of possible cognitive states."""
    EXPLORING = "exploring"
    EXPLOITING = "exploiting"
    REFLECTING = "reflecting"
    ADAPTING = "adapting"
    STUCK = "stuck"
    CONVERGING = "converging"


class MetaStrategy(Enum):
    """Meta-level cognitive strategies."""
    BREADTH_FIRST = "breadth_first"
    DEPTH_FIRST = "depth_first"
    BEST_FIRST = "best_first"
    RANDOM_WALK = "random_walk"
    GRADIENT_ASCENT = "gradient_ascent"
    SIMULATED_ANNEALING = "simulated_annealing"


@dataclass
class CognitiveMetrics:
    """Metrics for monitoring cognitive performance."""
    accuracy: float = 0.0
    efficiency: float = 0.0
    novelty: float = 0.0
    coherence: float = 0.0
    convergence_rate: float = 0.0
    resource_utilization: float = 0.0
    synergy_index: float = 0.0
    meta_confidence: float = 0.0
    
    def overall_performance(self) -> float:
        """Calculate overall performance score."""
        weights = {
            'accuracy': 0.25,
            'efficiency': 0.20,
            'novelty': 0.15,
            'coherence': 0.15,
            'convergence_rate': 0.10,
            'resource_utilization': 0.10,
            'synergy_index': 0.05
        }
        
        score = (
            self.accuracy * weights['accuracy'] +
            self.efficiency * weights['efficiency'] +
            self.novelty * weights['novelty'] +
            self.coherence * weights['coherence'] +
            self.convergence_rate * weights['convergence_rate'] +
            self.resource_utilization * weights['resource_utilization'] +
            self.synergy_index * weights['synergy_index']
        )
        
        return min(max(score, 0.0), 1.0)


@dataclass
class MetaCognitiveState:
    """Represents the current meta-cognitive state of the system."""
    current_state: CognitiveState = CognitiveState.EXPLORING
    active_strategy: MetaStrategy = MetaStrategy.BREADTH_FIRST
    confidence_level: float = 0.5
    attention_focus: Set[str] = field(default_factory=set)
    recent_performance: List[float] = field(default_factory=list)
    strategy_history: List[Tuple[MetaStrategy, float]] = field(default_factory=list)
    last_adaptation: datetime = field(default_factory=datetime.now)
    
    def update_performance(self, performance: float):
        """Update recent performance history."""
        self.recent_performance.append(performance)
        if len(self.recent_performance) > 20:  # Keep last 20 measurements
            self.recent_performance.pop(0)
    
    def get_performance_trend(self) -> str:
        """Analyze performance trend."""
        if len(self.recent_performance) < 3:
            return "insufficient_data"
        
        recent = self.recent_performance[-3:]
        if all(recent[i] > recent[i-1] for i in range(1, len(recent))):
            return "improving"
        elif all(recent[i] < recent[i-1] for i in range(1, len(recent))):
            return "declining"
        else:
            return "stable"


class MetaCognitiveMonitor:
    """
    Monitors cognitive processes and maintains meta-cognitive awareness.
    """
    
    def __init__(self, synergy_engine: CognitiveSynergyEngine):
        self.synergy_engine = synergy_engine
        self.meta_state = MetaCognitiveState()
        self.performance_history = deque(maxlen=1000)
        self.strategy_effectiveness = defaultdict(list)
        self.cognitive_patterns = {}
        self.monitoring_active = False
        
    def start_monitoring(self):
        """Start meta-cognitive monitoring."""
        self.monitoring_active = True
        logger.info("Meta-cognitive monitoring started")
        
        # Start monitoring thread
        monitoring_thread = threading.Thread(target=self._monitoring_loop)
        monitoring_thread.daemon = True
        monitoring_thread.start()
    
    def stop_monitoring(self):
        """Stop meta-cognitive monitoring."""
        self.monitoring_active = False
        logger.info("Meta-cognitive monitoring stopped")
    
    def _monitoring_loop(self):
        """Main monitoring loop."""
        while self.monitoring_active:
            try:
                # Collect current metrics
                metrics = self._collect_cognitive_metrics()
                
                # Update meta-cognitive state
                self._update_meta_state(metrics)
                
                # Detect cognitive patterns
                self._detect_cognitive_patterns()
                
                # Check if adaptation is needed
                if self._should_adapt():
                    self._trigger_adaptation()
                
                # Sleep before next iteration
                threading.Event().wait(1.0)  # 1 second intervals
                
            except Exception as e:
                logger.error(f"Error in meta-cognitive monitoring: {e}")
    
    def _collect_cognitive_metrics(self) -> CognitiveMetrics:
        """Collect current cognitive performance metrics."""
        metrics = CognitiveMetrics()
        
        # Get synergy metrics from engine
        synergy_metrics = self.synergy_engine.get_synergy_metrics()
        
        # Calculate accuracy based on process performance
        process_performances = []
        processes = self.synergy_engine.processes.read().unwrap() if hasattr(self.synergy_engine.processes, 'read') else {}
        
        for process in processes.values():
            if hasattr(process, 'performance_metrics') and process.performance_metrics:
                avg_perf = statistics.mean(process.performance_metrics.values())
                process_performances.append(avg_perf)
        
        if process_performances:
            metrics.accuracy = statistics.mean(process_performances)
        
        # Calculate efficiency based on resource utilization
        metrics.efficiency = synergy_metrics.get('process_efficiency', 0.5)
        
        # Calculate novelty based on recent pattern discoveries
        memory = self.synergy_engine.memory
        high_attention_atoms = memory.get_high_attention_atoms(0.3)
        metrics.novelty = min(len(high_attention_atoms) / 10.0, 1.0)  # Normalize
        
        # Calculate coherence based on attention distribution
        metrics.coherence = synergy_metrics.get('attention_distribution', 0.0) / 10.0
        
        # Calculate synergy index
        metrics.synergy_index = synergy_metrics.get('process_efficiency', 0.0)
        
        # Calculate meta-confidence based on recent performance stability
        if len(self.meta_state.recent_performance) > 5:
            variance = statistics.variance(self.meta_state.recent_performance[-5:])
            metrics.meta_confidence = max(0.0, 1.0 - variance)
        
        return metrics
    
    def _update_meta_state(self, metrics: CognitiveMetrics):
        """Update meta-cognitive state based on current metrics."""
        overall_performance = metrics.overall_performance()
        self.meta_state.update_performance(overall_performance)
        
        # Update confidence level
        trend = self.meta_state.get_performance_trend()
        if trend == "improving":
            self.meta_state.confidence_level = min(1.0, self.meta_state.confidence_level + 0.1)
        elif trend == "declining":
            self.meta_state.confidence_level = max(0.0, self.meta_state.confidence_level - 0.1)
        
        # Determine cognitive state
        if overall_performance < 0.3:
            self.meta_state.current_state = CognitiveState.STUCK
        elif trend == "improving" and overall_performance > 0.7:
            self.meta_state.current_state = CognitiveState.CONVERGING
        elif metrics.novelty > 0.6:
            self.meta_state.current_state = CognitiveState.EXPLORING
        elif metrics.efficiency > 0.7:
            self.meta_state.current_state = CognitiveState.EXPLOITING
        else:
            self.meta_state.current_state = CognitiveState.REFLECTING
    
    def _detect_cognitive_patterns(self):
        """Detect patterns in cognitive behavior."""
        if len(self.performance_history) < 10:
            return
        
        recent_performance = list(self.performance_history)[-10:]
        
        # Detect oscillation patterns
        peaks = []
        valleys = []
        for i in range(1, len(recent_performance) - 1):
            if (recent_performance[i] > recent_performance[i-1] and 
                recent_performance[i] > recent_performance[i+1]):
                peaks.append(i)
            elif (recent_performance[i] < recent_performance[i-1] and 
                  recent_performance[i] < recent_performance[i+1]):
                valleys.append(i)
        
        if len(peaks) > 2 and len(valleys) > 2:
            self.cognitive_patterns['oscillation'] = True
        else:
            self.cognitive_patterns['oscillation'] = False
        
        # Detect plateau patterns
        recent_variance = statistics.variance(recent_performance)
        if recent_variance < 0.01:  # Very low variance indicates plateau
            self.cognitive_patterns['plateau'] = True
        else:
            self.cognitive_patterns['plateau'] = False
    
    def _should_adapt(self) -> bool:
        """Determine if cognitive adaptation is needed."""
        # Adapt if performance is declining
        if self.meta_state.get_performance_trend() == "declining":
            return True
        
        # Adapt if stuck in a plateau
        if self.cognitive_patterns.get('plateau', False):
            return True
        
        # Adapt if confidence is very low
        if self.meta_state.confidence_level < 0.2:
            return True
        
        # Adapt if it's been too long since last adaptation
        time_since_adaptation = datetime.now() - self.meta_state.last_adaptation
        if time_since_adaptation > timedelta(minutes=5):
            return True
        
        return False
    
    def _trigger_adaptation(self):
        """Trigger cognitive adaptation."""
        logger.info("Triggering cognitive adaptation")
        
        # Record current strategy effectiveness
        current_performance = self.meta_state.recent_performance[-1] if self.meta_state.recent_performance else 0.5
        self.strategy_effectiveness[self.meta_state.active_strategy].append(current_performance)
        
        # Choose new strategy based on effectiveness history
        new_strategy = self._select_optimal_strategy()
        
        # Update meta-state
        self.meta_state.active_strategy = new_strategy
        self.meta_state.last_adaptation = datetime.now()
        self.meta_state.strategy_history.append((new_strategy, current_performance))
        
        # Apply the new strategy
        self._apply_strategy(new_strategy)
    
    def _select_optimal_strategy(self) -> MetaStrategy:
        """Select the optimal strategy based on historical effectiveness."""
        # If we don't have enough data, explore randomly
        if sum(len(performances) for performances in self.strategy_effectiveness.values()) < 10:
            strategies = list(MetaStrategy)
            return np.random.choice(strategies)
        
        # Calculate average effectiveness for each strategy
        strategy_scores = {}
        for strategy, performances in self.strategy_effectiveness.items():
            if performances:
                strategy_scores[strategy] = statistics.mean(performances)
        
        # Select strategy with highest average performance
        if strategy_scores:
            best_strategy = max(strategy_scores.keys(), key=lambda k: strategy_scores[k])
            
            # Add some exploration: 20% chance to try a different strategy
            if np.random.random() < 0.2:
                strategies = list(MetaStrategy)
                strategies.remove(best_strategy)
                return np.random.choice(strategies)
            
            return best_strategy
        
        # Fallback to breadth-first
        return MetaStrategy.BREADTH_FIRST
    
    def _apply_strategy(self, strategy: MetaStrategy):
        """Apply the selected cognitive strategy."""
        logger.info(f"Applying cognitive strategy: {strategy.value}")
        
        # Strategy-specific implementations
        if strategy == MetaStrategy.BREADTH_FIRST:
            self._apply_breadth_first_strategy()
        elif strategy == MetaStrategy.DEPTH_FIRST:
            self._apply_depth_first_strategy()
        elif strategy == MetaStrategy.BEST_FIRST:
            self._apply_best_first_strategy()
        elif strategy == MetaStrategy.RANDOM_WALK:
            self._apply_random_walk_strategy()
        elif strategy == MetaStrategy.GRADIENT_ASCENT:
            self._apply_gradient_ascent_strategy()
        elif strategy == MetaStrategy.SIMULATED_ANNEALING:
            self._apply_simulated_annealing_strategy()
    
    def _apply_breadth_first_strategy(self):
        """Apply breadth-first exploration strategy."""
        # Distribute attention evenly across all high-attention atoms
        memory = self.synergy_engine.memory
        high_attention_atoms = memory.get_high_attention_atoms(0.3)
        
        attention_boost = 0.1 / max(len(high_attention_atoms), 1)
        for atom_id in high_attention_atoms:
            memory.update_attention(atom_id, attention_boost)
    
    def _apply_depth_first_strategy(self):
        """Apply depth-first exploration strategy."""
        # Focus attention on the highest-attention atom and its neighbors
        memory = self.synergy_engine.memory
        high_attention_atoms = memory.get_high_attention_atoms(0.5)
        
        if high_attention_atoms:
            focus_atom = high_attention_atoms[0]  # Highest attention
            neighbors = memory.get_neighbors(focus_atom)
            
            # Boost attention for focus atom and neighbors
            memory.update_attention(focus_atom, 0.2)
            for neighbor in neighbors:
                memory.update_attention(neighbor, 0.1)
    
    def _apply_best_first_strategy(self):
        """Apply best-first search strategy."""
        # Focus on atoms with highest truth values and attention
        memory = self.synergy_engine.memory
        atoms = memory.atoms
        
        if hasattr(atoms, 'read'):
            atoms_dict = atoms.read().unwrap()
        else:
            atoms_dict = atoms
        
        # Score atoms by truth_value * attention_value
        atom_scores = []
        for atom_id, atom in atoms_dict.items():
            score = atom.truth_value * atom.attention_value
            atom_scores.append((atom_id, score))
        
        # Sort by score and boost top atoms
        atom_scores.sort(key=lambda x: x[1], reverse=True)
        for atom_id, score in atom_scores[:5]:  # Top 5 atoms
            memory.update_attention(atom_id, 0.1)
    
    def _apply_random_walk_strategy(self):
        """Apply random walk exploration strategy."""
        # Randomly select atoms and boost their attention
        memory = self.synergy_engine.memory
        atoms = memory.atoms
        
        if hasattr(atoms, 'read'):
            atoms_dict = atoms.read().unwrap()
        else:
            atoms_dict = atoms
        
        atom_ids = list(atoms_dict.keys())
        if atom_ids:
            # Randomly select 3 atoms
            selected_atoms = np.random.choice(atom_ids, size=min(3, len(atom_ids)), replace=False)
            for atom_id in selected_atoms:
                memory.update_attention(atom_id, 0.15)
    
    def _apply_gradient_ascent_strategy(self):
        """Apply gradient ascent optimization strategy."""
        # Boost attention for atoms that have been increasing in attention
        memory = self.synergy_engine.memory
        
        # This would require tracking attention changes over time
        # For now, boost atoms with above-average attention
        high_attention_atoms = memory.get_high_attention_atoms(0.4)
        for atom_id in high_attention_atoms:
            memory.update_attention(atom_id, 0.05)
    
    def _apply_simulated_annealing_strategy(self):
        """Apply simulated annealing strategy."""
        # Combine random exploration with gradual focusing
        temperature = max(0.1, 1.0 - (len(self.performance_history) / 1000.0))
        
        memory = self.synergy_engine.memory
        atoms = memory.atoms
        
        if hasattr(atoms, 'read'):
            atoms_dict = atoms.read().unwrap()
        else:
            atoms_dict = atoms
        
        atom_ids = list(atoms_dict.keys())
        if atom_ids:
            # Higher temperature = more random, lower temperature = more focused
            num_atoms_to_boost = max(1, int(len(atom_ids) * temperature))
            selected_atoms = np.random.choice(atom_ids, size=num_atoms_to_boost, replace=False)
            
            boost_amount = 0.1 * (1.0 - temperature)  # Less boost at higher temperature
            for atom_id in selected_atoms:
                memory.update_attention(atom_id, boost_amount)


class SelfReflectiveLearner:
    """
    Implements self-reflective learning mechanisms for cognitive improvement.
    """
    
    def __init__(self, monitor: MetaCognitiveMonitor):
        self.monitor = monitor
        self.learning_history = []
        self.cognitive_models = {}
        self.reflection_insights = []
        
    def reflect_on_performance(self) -> Dict[str, Any]:
        """Perform self-reflection on recent cognitive performance."""
        insights = {
            'timestamp': datetime.now().isoformat(),
            'performance_analysis': self._analyze_performance(),
            'strategy_effectiveness': self._analyze_strategy_effectiveness(),
            'pattern_insights': self._analyze_cognitive_patterns(),
            'recommendations': self._generate_recommendations()
        }
        
        self.reflection_insights.append(insights)
        return insights
    
    def _analyze_performance(self) -> Dict[str, Any]:
        """Analyze recent performance trends."""
        recent_performance = self.monitor.meta_state.recent_performance
        
        if len(recent_performance) < 5:
            return {'status': 'insufficient_data'}
        
        analysis = {
            'mean_performance': statistics.mean(recent_performance),
            'performance_variance': statistics.variance(recent_performance),
            'trend': self.monitor.meta_state.get_performance_trend(),
            'stability': 1.0 - statistics.variance(recent_performance[-5:]),
            'peak_performance': max(recent_performance),
            'lowest_performance': min(recent_performance)
        }
        
        return analysis
    
    def _analyze_strategy_effectiveness(self) -> Dict[str, Any]:
        """Analyze the effectiveness of different cognitive strategies."""
        effectiveness = {}
        
        for strategy, performances in self.monitor.strategy_effectiveness.items():
            if performances:
                effectiveness[strategy.value] = {
                    'mean_performance': statistics.mean(performances),
                    'consistency': 1.0 - statistics.variance(performances),
                    'usage_count': len(performances),
                    'best_performance': max(performances),
                    'worst_performance': min(performances)
                }
        
        return effectiveness
    
    def _analyze_cognitive_patterns(self) -> Dict[str, Any]:
        """Analyze detected cognitive patterns."""
        patterns = self.monitor.cognitive_patterns.copy()
        
        # Add temporal analysis
        if len(self.monitor.performance_history) > 20:
            recent_data = list(self.monitor.performance_history)[-20:]
            
            # Detect cycles
            autocorr = np.correlate(recent_data, recent_data, mode='full')
            patterns['autocorrelation_peak'] = np.argmax(autocorr[len(autocorr)//2:])
            
            # Detect trends
            x = np.arange(len(recent_data))
            slope, _ = np.polyfit(x, recent_data, 1)
            patterns['trend_slope'] = slope
        
        return patterns
    
    def _generate_recommendations(self) -> List[str]:
        """Generate recommendations for cognitive improvement."""
        recommendations = []
        
        # Analyze current state
        current_state = self.monitor.meta_state.current_state
        confidence = self.monitor.meta_state.confidence_level
        trend = self.monitor.meta_state.get_performance_trend()
        
        if current_state == CognitiveState.STUCK:
            recommendations.append("Consider switching to random walk strategy to escape local optima")
            recommendations.append("Increase exploration by boosting attention to novel patterns")
        
        if confidence < 0.3:
            recommendations.append("Reduce cognitive load by focusing on fewer high-priority tasks")
            recommendations.append("Implement more conservative strategies until confidence improves")
        
        if trend == "declining":
            recommendations.append("Analyze recent strategy changes and revert if necessary")
            recommendations.append("Increase meta-cognitive monitoring frequency")
        
        if self.monitor.cognitive_patterns.get('oscillation', False):
            recommendations.append("Implement damping mechanisms to reduce oscillatory behavior")
            recommendations.append("Consider longer-term strategy commitments")
        
        if self.monitor.cognitive_patterns.get('plateau', False):
            recommendations.append("Introduce novelty through random exploration")
            recommendations.append("Consider meta-strategy adaptation")
        
        return recommendations
    
    def learn_from_reflection(self, insights: Dict[str, Any]):
        """Learn from reflection insights to improve future performance."""
        # Update cognitive models based on insights
        self._update_cognitive_models(insights)
        
        # Adjust meta-cognitive parameters
        self._adjust_metacognitive_parameters(insights)
        
        # Store learning for future reference
        self.learning_history.append({
            'timestamp': datetime.now().isoformat(),
            'insights': insights,
            'actions_taken': self._get_actions_taken(insights)
        })
    
    def _update_cognitive_models(self, insights: Dict[str, Any]):
        """Update internal models of cognitive processes."""
        # Update strategy effectiveness model
        strategy_data = insights.get('strategy_effectiveness', {})
        for strategy, data in strategy_data.items():
            if strategy not in self.cognitive_models:
                self.cognitive_models[strategy] = {
                    'expected_performance': 0.5,
                    'confidence': 0.5,
                    'usage_contexts': []
                }
            
            # Update expected performance with exponential moving average
            alpha = 0.1  # Learning rate
            current_expected = self.cognitive_models[strategy]['expected_performance']
            new_performance = data['mean_performance']
            self.cognitive_models[strategy]['expected_performance'] = (
                alpha * new_performance + (1 - alpha) * current_expected
            )
    
    def _adjust_metacognitive_parameters(self, insights: Dict[str, Any]):
        """Adjust meta-cognitive parameters based on insights."""
        performance_analysis = insights.get('performance_analysis', {})
        
        # Adjust adaptation threshold based on performance stability
        stability = performance_analysis.get('stability', 0.5)
        if stability < 0.3:
            # High instability - adapt more frequently
            self.monitor.meta_state.confidence_level *= 0.9
        elif stability > 0.8:
            # High stability - can be more confident
            self.monitor.meta_state.confidence_level = min(1.0, self.monitor.meta_state.confidence_level * 1.1)
    
    def _get_actions_taken(self, insights: Dict[str, Any]) -> List[str]:
        """Record actions taken based on insights."""
        actions = []
        
        recommendations = insights.get('recommendations', [])
        for rec in recommendations:
            if "random walk" in rec.lower():
                actions.append("switched_to_random_walk_strategy")
            elif "reduce cognitive load" in rec.lower():
                actions.append("reduced_attention_distribution")
            elif "increase exploration" in rec.lower():
                actions.append("increased_novelty_seeking")
        
        return actions


def demonstrate_meta_cognitive_reasoning():
    """
    Demonstrate the meta-cognitive reasoning capabilities.
    """
    print("=== Meta-Cognitive Reasoning Demonstration ===\n")
    
    # Create synergy engine
    from cognitive_synergy_framework import CognitiveSynergyEngine, CognitiveProcess
    engine = CognitiveSynergyEngine()
    
    # Register some cognitive processes
    reasoning_process = CognitiveProcess(
        process_id="meta_reasoning",
        process_type="meta_cognitive",
        priority=0.9
    )
    reasoning_process.performance_metrics = {'accuracy': 0.7, 'speed': 0.8}
    
    learning_process = CognitiveProcess(
        process_id="adaptive_learning",
        process_type="machine_learning",
        priority=0.8
    )
    learning_process.performance_metrics = {'accuracy': 0.6, 'novelty': 0.9}
    
    engine.register_process(reasoning_process)
    engine.register_process(learning_process)
    
    # Add some atoms to memory
    memory = engine.get_memory()
    concept1 = Atom(atom_type="ConceptNode", name="meta_cognition", truth_value=0.9, attention_value=0.8)
    concept2 = Atom(atom_type="ConceptNode", name="self_reflection", truth_value=0.8, attention_value=0.7)
    concept3 = Atom(atom_type="ConceptNode", name="adaptation", truth_value=0.7, attention_value=0.6)
    
    memory.add_atom(concept1)
    memory.add_atom(concept2)
    memory.add_atom(concept3)
    
    # Create meta-cognitive monitor
    monitor = MetaCognitiveMonitor(engine)
    
    # Create self-reflective learner
    learner = SelfReflectiveLearner(monitor)
    
    print("Initial meta-cognitive state:")
    print(f"  Current state: {monitor.meta_state.current_state.value}")
    print(f"  Active strategy: {monitor.meta_state.active_strategy.value}")
    print(f"  Confidence level: {monitor.meta_state.confidence_level:.3f}")
    
    # Start monitoring
    monitor.start_monitoring()
    
    # Simulate some cognitive activity
    print("\nSimulating cognitive activity...")
    import time
    
    for i in range(10):
        # Simulate varying performance
        performance = 0.5 + 0.3 * np.sin(i * 0.5) + 0.1 * np.random.random()
        monitor.meta_state.update_performance(performance)
        
        # Update process performance
        reasoning_process.performance_metrics['accuracy'] = performance
        learning_process.performance_metrics['accuracy'] = performance * 0.9
        
        time.sleep(0.5)  # Brief pause
    
    # Perform reflection
    print("\nPerforming self-reflection...")
    insights = learner.reflect_on_performance()
    
    print(f"Performance Analysis:")
    perf_analysis = insights['performance_analysis']
    if 'mean_performance' in perf_analysis:
        print(f"  Mean performance: {perf_analysis['mean_performance']:.3f}")
        print(f"  Performance trend: {perf_analysis['trend']}")
        print(f"  Stability: {perf_analysis['stability']:.3f}")
    
    print(f"\nStrategy Effectiveness:")
    strategy_eff = insights['strategy_effectiveness']
    for strategy, data in strategy_eff.items():
        print(f"  {strategy}: {data['mean_performance']:.3f} (used {data['usage_count']} times)")
    
    print(f"\nRecommendations:")
    for rec in insights['recommendations']:
        print(f"  - {rec}")
    
    # Learn from reflection
    learner.learn_from_reflection(insights)
    
    print(f"\nFinal meta-cognitive state:")
    print(f"  Current state: {monitor.meta_state.current_state.value}")
    print(f"  Active strategy: {monitor.meta_state.active_strategy.value}")
    print(f"  Confidence level: {monitor.meta_state.confidence_level:.3f}")
    print(f"  Performance trend: {monitor.meta_state.get_performance_trend()}")
    
    # Stop monitoring
    monitor.stop_monitoring()
    
    print("\n=== Meta-Cognitive Reasoning Demo Complete ===")


if __name__ == "__main__":
    demonstrate_meta_cognitive_reasoning()

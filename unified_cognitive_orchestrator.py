#!/usr/bin/env python3
"""
Unified Cognitive Orchestrator for OpenCog Collection
=====================================================

This module provides a master orchestration layer that coordinates all cognitive
processes for true cognitive synergy. It implements:

1. Multi-process coordination and attention allocation
2. Bottleneck detection and dynamic resource reallocation
3. Cross-module pattern propagation
4. Emergent behavior tracking and amplification
5. Synergy metrics collection and reporting

Based on:
- Ben Goertzel's formal model of cognitive synergy (arXiv:1703.04361)
- Deep Tree Echo membrane architecture
- Agent-Arena-Relation (AAR) geometric self-awareness

Author: OpenCog Collection Contributors
License: GPL-3.0+
"""

import logging
import threading
import time
from typing import Dict, List, Any, Optional, Set, Tuple, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict, deque
from queue import PriorityQueue, Queue
from enum import Enum
import json
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ProcessState(Enum):
    """States a cognitive process can be in."""
    IDLE = "idle"
    ACTIVE = "active"
    BLOCKED = "blocked"
    STUCK = "stuck"
    COMPLETED = "completed"


class AttentionPriority(Enum):
    """Priority levels for attention allocation."""
    CRITICAL = 5
    HIGH = 4
    NORMAL = 3
    LOW = 2
    BACKGROUND = 1


@dataclass
class CognitiveProcess:
    """Represents a cognitive process in the architecture."""
    process_id: str
    process_type: str
    state: ProcessState = ProcessState.IDLE
    priority: AttentionPriority = AttentionPriority.NORMAL
    attention_allocated: float = 0.0
    attention_requested: float = 0.0
    patterns_discovered: int = 0
    patterns_consumed: int = 0
    bottleneck_count: int = 0
    last_activity: datetime = field(default_factory=datetime.now)
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    dependencies: Set[str] = field(default_factory=set)
    outputs: List[Any] = field(default_factory=list)
    
    def update_metrics(self, metric_name: str, value: float):
        """Update performance metrics."""
        self.performance_metrics[metric_name] = value
        self.last_activity = datetime.now()


@dataclass
class Pattern:
    """Represents a pattern discovered by a cognitive process."""
    pattern_id: str
    source_process: str
    pattern_type: str
    content: Any
    confidence: float
    attention_value: float
    timestamp: datetime = field(default_factory=datetime.now)
    consumers: Set[str] = field(default_factory=set)
    
    def __hash__(self):
        return hash(self.pattern_id)


@dataclass
class SynergyMetrics:
    """Metrics tracking cognitive synergy."""
    cross_module_patterns: int = 0
    bottlenecks_resolved: int = 0
    emergent_behaviors: int = 0
    attention_efficiency: float = 0.0
    knowledge_integration_rate: float = 0.0
    total_patterns_shared: int = 0
    active_processes: int = 0
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary."""
        return {
            "cross_module_patterns": self.cross_module_patterns,
            "bottlenecks_resolved": self.bottlenecks_resolved,
            "emergent_behaviors": self.emergent_behaviors,
            "attention_efficiency": self.attention_efficiency,
            "knowledge_integration_rate": self.knowledge_integration_rate,
            "total_patterns_shared": self.total_patterns_shared,
            "active_processes": self.active_processes,
            "timestamp": self.timestamp.isoformat()
        }


class HypergraphKnowledgeBase:
    """
    Shared hypergraph knowledge base for all cognitive processes.
    Implements the AtomSpace-like interface for pattern storage and retrieval.
    """
    
    def __init__(self):
        self.patterns: Dict[str, Pattern] = {}
        self.pattern_index: Dict[str, Set[str]] = defaultdict(set)
        self.attention_bank: Dict[str, float] = defaultdict(float)
        self.lock = threading.RLock()
        
    def add_pattern(self, pattern: Pattern) -> str:
        """Add a pattern to the knowledge base."""
        with self.lock:
            self.patterns[pattern.pattern_id] = pattern
            self.pattern_index[pattern.pattern_type].add(pattern.pattern_id)
            self.attention_bank[pattern.pattern_id] = pattern.attention_value
            logger.debug(f"Added pattern {pattern.pattern_id} from {pattern.source_process}")
            return pattern.pattern_id
    
    def get_patterns_by_type(self, pattern_type: str) -> List[Pattern]:
        """Retrieve all patterns of a specific type."""
        with self.lock:
            pattern_ids = self.pattern_index.get(pattern_type, set())
            return [self.patterns[pid] for pid in pattern_ids if pid in self.patterns]
    
    def get_high_attention_patterns(self, threshold: float = 0.5) -> List[Pattern]:
        """Get patterns with attention value above threshold."""
        with self.lock:
            return [p for p in self.patterns.values() if p.attention_value >= threshold]
    
    def propagate_attention(self, pattern_id: str, delta: float):
        """Propagate attention to a pattern."""
        with self.lock:
            if pattern_id in self.patterns:
                self.patterns[pattern_id].attention_value += delta
                self.attention_bank[pattern_id] += delta


class UnifiedCognitiveOrchestrator:
    """
    Master orchestrator coordinating all cognitive processes for synergy.
    
    This orchestrator implements:
    - Attention economy: Manages attention as a scarce resource
    - Pattern propagation: Shares discoveries across all processes
    - Bottleneck resolution: Detects and routes around stuck processes
    - Emergent behavior detection: Identifies novel behaviors from interactions
    """
    
    def __init__(self, total_attention: float = 100.0):
        """
        Initialize the cognitive orchestrator.
        
        Args:
            total_attention: Total attention budget to allocate across processes
        """
        self.processes: Dict[str, CognitiveProcess] = {}
        self.knowledge_base = HypergraphKnowledgeBase()
        self.total_attention = total_attention
        self.available_attention = total_attention
        self.metrics = SynergyMetrics()
        
        # Queues for coordination
        self.pattern_queue: Queue = Queue()
        self.attention_requests: PriorityQueue = PriorityQueue()
        
        # History tracking
        self.pattern_history: deque = deque(maxlen=1000)
        self.bottleneck_history: deque = deque(maxlen=100)
        self.emergent_behaviors: List[Dict[str, Any]] = []
        
        # Control
        self.running = False
        self.orchestration_thread: Optional[threading.Thread] = None
        self.lock = threading.RLock()
        
        logger.info(f"Unified Cognitive Orchestrator initialized with {total_attention} attention units")
    
    def register_process(self, process: CognitiveProcess):
        """Register a cognitive process with the orchestrator."""
        with self.lock:
            self.processes[process.process_id] = process
            logger.info(f"Registered process: {process.process_id} ({process.process_type})")
    
    def unregister_process(self, process_id: str):
        """Unregister a cognitive process."""
        with self.lock:
            if process_id in self.processes:
                del self.processes[process_id]
                logger.info(f"Unregistered process: {process_id}")
    
    def request_attention(self, process_id: str, amount: float, priority: AttentionPriority):
        """
        Request attention allocation for a process.
        
        Args:
            process_id: ID of the requesting process
            amount: Amount of attention requested
            priority: Priority level of the request
        """
        if process_id in self.processes:
            self.processes[process_id].attention_requested = amount
            # Use negative priority value for PriorityQueue (lower value = higher priority)
            self.attention_requests.put((-priority.value, process_id, amount))
            logger.debug(f"Process {process_id} requested {amount} attention at {priority.name} priority")
    
    def allocate_attention(self):
        """
        Allocate attention to processes based on priority and availability.
        Implements attention economy principles.
        """
        with self.lock:
            # Reset allocations
            for process in self.processes.values():
                process.attention_allocated = 0.0
            
            self.available_attention = self.total_attention
            allocated_processes = []
            
            # Process requests in priority order
            while not self.attention_requests.empty() and self.available_attention > 0:
                neg_priority, process_id, amount = self.attention_requests.get()
                
                if process_id not in self.processes:
                    continue
                
                process = self.processes[process_id]
                
                # Allocate available attention
                allocated = min(amount, self.available_attention)
                process.attention_allocated = allocated
                self.available_attention -= allocated
                allocated_processes.append((process_id, allocated))
                
                logger.debug(f"Allocated {allocated} attention to {process_id}")
            
            # Update metrics
            if self.total_attention > 0:
                self.metrics.attention_efficiency = (
                    (self.total_attention - self.available_attention) / self.total_attention
                )
            
            return allocated_processes
    
    def submit_pattern(self, pattern: Pattern):
        """
        Submit a discovered pattern for sharing across processes.
        
        Args:
            pattern: The pattern to share
        """
        pattern_id = self.knowledge_base.add_pattern(pattern)
        self.pattern_queue.put(pattern)
        self.pattern_history.append(pattern)
        
        # Update process metrics
        if pattern.source_process in self.processes:
            self.processes[pattern.source_process].patterns_discovered += 1
        
        logger.info(f"Pattern {pattern_id} submitted by {pattern.source_process}")
    
    def propagate_patterns(self):
        """
        Propagate patterns to relevant processes.
        Implements cross-module pattern sharing for synergy.
        """
        patterns_propagated = 0
        
        while not self.pattern_queue.empty():
            pattern = self.pattern_queue.get()
            
            # Find processes that can consume this pattern
            for process_id, process in self.processes.items():
                # Don't send pattern back to source
                if process_id == pattern.source_process:
                    continue
                
                # Check if process can use this pattern type
                # (In real implementation, this would check process capabilities)
                if self._can_consume_pattern(process, pattern):
                    pattern.consumers.add(process_id)
                    process.patterns_consumed += 1
                    process.outputs.append(pattern)
                    patterns_propagated += 1
                    
                    logger.debug(f"Propagated pattern {pattern.pattern_id} to {process_id}")
        
        if patterns_propagated > 0:
            self.metrics.cross_module_patterns += patterns_propagated
            self.metrics.total_patterns_shared += patterns_propagated
    
    def _can_consume_pattern(self, process: CognitiveProcess, pattern: Pattern) -> bool:
        """
        Determine if a process can consume a pattern.
        
        Args:
            process: The cognitive process
            pattern: The pattern to check
            
        Returns:
            True if the process can consume the pattern
        """
        # Simple heuristic: processes can consume patterns from different types
        # In real implementation, this would be more sophisticated
        return process.process_type != pattern.source_process
    
    def detect_bottlenecks(self) -> List[str]:
        """
        Detect processes that are stuck or blocked.
        
        Returns:
            List of process IDs that are bottlenecked
        """
        bottlenecks = []
        current_time = datetime.now()
        
        with self.lock:
            for process_id, process in self.processes.items():
                # Check if process is stuck (no activity for too long)
                time_since_activity = current_time - process.last_activity
                
                if time_since_activity > timedelta(seconds=30):
                    if process.state == ProcessState.ACTIVE:
                        process.state = ProcessState.STUCK
                        process.bottleneck_count += 1
                        bottlenecks.append(process_id)
                        self.bottleneck_history.append({
                            "process_id": process_id,
                            "timestamp": current_time,
                            "time_stuck": time_since_activity.total_seconds()
                        })
                        logger.warning(f"Bottleneck detected in process {process_id}")
        
        return bottlenecks
    
    def resolve_bottlenecks(self, bottlenecks: List[str]):
        """
        Attempt to resolve bottlenecks by reallocating resources.
        
        Args:
            bottlenecks: List of bottlenecked process IDs
        """
        for process_id in bottlenecks:
            if process_id in self.processes:
                process = self.processes[process_id]
                
                # Strategy 1: Boost attention allocation
                self.request_attention(process_id, process.attention_requested * 1.5, 
                                     AttentionPriority.HIGH)
                
                # Strategy 2: Share relevant patterns
                relevant_patterns = self.knowledge_base.get_high_attention_patterns()
                for pattern in relevant_patterns[:5]:  # Top 5 patterns
                    if self._can_consume_pattern(process, pattern):
                        process.outputs.append(pattern)
                        process.patterns_consumed += 1
                
                logger.info(f"Attempted bottleneck resolution for {process_id}")
                self.metrics.bottlenecks_resolved += 1
    
    def detect_emergent_behaviors(self):
        """
        Detect emergent behaviors arising from process interactions.
        
        Emergent behaviors are identified when:
        1. Multiple processes contribute to a pattern
        2. Pattern has high attention value
        3. Pattern is consumed by many processes
        """
        high_attention_patterns = self.knowledge_base.get_high_attention_patterns(threshold=0.7)
        
        for pattern in high_attention_patterns:
            # Check if pattern has multiple consumers (synergy indicator)
            if len(pattern.consumers) >= 3:
                emergent_behavior = {
                    "pattern_id": pattern.pattern_id,
                    "source": pattern.source_process,
                    "consumers": list(pattern.consumers),
                    "attention": pattern.attention_value,
                    "timestamp": datetime.now().isoformat()
                }
                
                # Check if this is a new emergent behavior
                if not any(eb["pattern_id"] == pattern.pattern_id for eb in self.emergent_behaviors):
                    self.emergent_behaviors.append(emergent_behavior)
                    self.metrics.emergent_behaviors += 1
                    logger.info(f"Emergent behavior detected: {pattern.pattern_id} "
                              f"consumed by {len(pattern.consumers)} processes")
    
    def orchestration_loop(self):
        """Main orchestration loop running in background thread."""
        logger.info("Orchestration loop started")
        
        while self.running:
            try:
                # 1. Allocate attention
                self.allocate_attention()
                
                # 2. Propagate patterns
                self.propagate_patterns()
                
                # 3. Detect bottlenecks
                bottlenecks = self.detect_bottlenecks()
                
                # 4. Resolve bottlenecks
                if bottlenecks:
                    self.resolve_bottlenecks(bottlenecks)
                
                # 5. Detect emergent behaviors
                self.detect_emergent_behaviors()
                
                # 6. Update metrics
                self.metrics.active_processes = sum(
                    1 for p in self.processes.values() 
                    if p.state == ProcessState.ACTIVE
                )
                self.metrics.timestamp = datetime.now()
                
                # Sleep briefly to avoid busy-waiting
                time.sleep(0.1)
                
            except Exception as e:
                logger.error(f"Error in orchestration loop: {e}", exc_info=True)
        
        logger.info("Orchestration loop stopped")
    
    def start(self):
        """Start the orchestrator."""
        if not self.running:
            self.running = True
            self.orchestration_thread = threading.Thread(target=self.orchestration_loop, daemon=True)
            self.orchestration_thread.start()
            logger.info("Unified Cognitive Orchestrator started")
    
    def stop(self):
        """Stop the orchestrator."""
        if self.running:
            self.running = False
            if self.orchestration_thread:
                self.orchestration_thread.join(timeout=5)
            logger.info("Unified Cognitive Orchestrator stopped")
    
    def get_metrics(self) -> SynergyMetrics:
        """Get current synergy metrics."""
        return self.metrics
    
    def get_status(self) -> Dict[str, Any]:
        """Get comprehensive status of the orchestrator."""
        with self.lock:
            return {
                "running": self.running,
                "total_processes": len(self.processes),
                "active_processes": sum(1 for p in self.processes.values() 
                                       if p.state == ProcessState.ACTIVE),
                "total_attention": self.total_attention,
                "available_attention": self.available_attention,
                "total_patterns": len(self.knowledge_base.patterns),
                "metrics": self.metrics.to_dict(),
                "processes": {
                    pid: {
                        "type": p.process_type,
                        "state": p.state.value,
                        "attention_allocated": p.attention_allocated,
                        "patterns_discovered": p.patterns_discovered,
                        "patterns_consumed": p.patterns_consumed,
                        "bottleneck_count": p.bottleneck_count
                    }
                    for pid, p in self.processes.items()
                }
            }
    
    def export_metrics(self, filepath: str):
        """Export metrics to JSON file."""
        status = self.get_status()
        with open(filepath, 'w') as f:
            json.dump(status, f, indent=2)
        logger.info(f"Metrics exported to {filepath}")


# Example usage
if __name__ == "__main__":
    # Create orchestrator
    orchestrator = UnifiedCognitiveOrchestrator(total_attention=100.0)
    
    # Register some example processes
    reasoning_process = CognitiveProcess(
        process_id="reasoning-1",
        process_type="reasoning",
        priority=AttentionPriority.HIGH
    )
    
    learning_process = CognitiveProcess(
        process_id="learning-1",
        process_type="learning",
        priority=AttentionPriority.NORMAL
    )
    
    perception_process = CognitiveProcess(
        process_id="perception-1",
        process_type="perception",
        priority=AttentionPriority.NORMAL
    )
    
    orchestrator.register_process(reasoning_process)
    orchestrator.register_process(learning_process)
    orchestrator.register_process(perception_process)
    
    # Start orchestrator
    orchestrator.start()
    
    # Simulate some activity
    orchestrator.request_attention("reasoning-1", 30.0, AttentionPriority.HIGH)
    orchestrator.request_attention("learning-1", 25.0, AttentionPriority.NORMAL)
    orchestrator.request_attention("perception-1", 20.0, AttentionPriority.NORMAL)
    
    # Submit some patterns
    pattern1 = Pattern(
        pattern_id="pattern-001",
        source_process="perception-1",
        pattern_type="visual",
        content={"type": "edge_detection", "confidence": 0.9},
        confidence=0.9,
        attention_value=0.8
    )
    
    orchestrator.submit_pattern(pattern1)
    
    # Let it run for a bit
    time.sleep(2)
    
    # Get status
    status = orchestrator.get_status()
    print(json.dumps(status, indent=2))
    
    # Stop orchestrator
    orchestrator.stop()

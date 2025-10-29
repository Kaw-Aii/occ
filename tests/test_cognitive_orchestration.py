#!/usr/bin/env python3
"""
Test Suite for Unified Cognitive Orchestrator
=============================================

Comprehensive tests for cognitive synergy orchestration including:
- Process registration and coordination
- Attention allocation mechanisms
- Pattern propagation across modules
- Bottleneck detection and resolution
- Emergent behavior identification
- Synergy metrics validation

Author: OpenCog Collection Contributors
License: GPL-3.0+
"""

import unittest
import time
import threading
from datetime import datetime, timedelta
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from unified_cognitive_orchestrator import (
    UnifiedCognitiveOrchestrator,
    CognitiveProcess,
    Pattern,
    ProcessState,
    AttentionPriority,
    SynergyMetrics
)


class TestCognitiveOrchestrator(unittest.TestCase):
    """Test cases for UnifiedCognitiveOrchestrator."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.orchestrator = UnifiedCognitiveOrchestrator(total_attention=100.0)
        
        # Create test processes
        self.reasoning_process = CognitiveProcess(
            process_id="test-reasoning",
            process_type="reasoning",
            priority=AttentionPriority.HIGH
        )
        
        self.learning_process = CognitiveProcess(
            process_id="test-learning",
            process_type="learning",
            priority=AttentionPriority.NORMAL
        )
        
        self.perception_process = CognitiveProcess(
            process_id="test-perception",
            process_type="perception",
            priority=AttentionPriority.NORMAL
        )
    
    def tearDown(self):
        """Clean up after tests."""
        if self.orchestrator.running:
            self.orchestrator.stop()
    
    def test_process_registration(self):
        """Test process registration and unregistration."""
        # Register process
        self.orchestrator.register_process(self.reasoning_process)
        self.assertIn("test-reasoning", self.orchestrator.processes)
        self.assertEqual(len(self.orchestrator.processes), 1)
        
        # Register another process
        self.orchestrator.register_process(self.learning_process)
        self.assertEqual(len(self.orchestrator.processes), 2)
        
        # Unregister process
        self.orchestrator.unregister_process("test-reasoning")
        self.assertNotIn("test-reasoning", self.orchestrator.processes)
        self.assertEqual(len(self.orchestrator.processes), 1)
    
    def test_attention_allocation(self):
        """Test attention allocation mechanism."""
        # Register processes
        self.orchestrator.register_process(self.reasoning_process)
        self.orchestrator.register_process(self.learning_process)
        
        # Request attention
        self.orchestrator.request_attention("test-reasoning", 40.0, AttentionPriority.HIGH)
        self.orchestrator.request_attention("test-learning", 30.0, AttentionPriority.NORMAL)
        
        # Allocate attention
        allocations = self.orchestrator.allocate_attention()
        
        # Verify allocations
        self.assertEqual(len(allocations), 2)
        
        # High priority should be allocated first
        reasoning_allocation = self.orchestrator.processes["test-reasoning"].attention_allocated
        learning_allocation = self.orchestrator.processes["test-learning"].attention_allocated
        
        self.assertEqual(reasoning_allocation, 40.0)
        self.assertEqual(learning_allocation, 30.0)
        self.assertEqual(self.orchestrator.available_attention, 30.0)
    
    def test_attention_scarcity(self):
        """Test attention allocation under scarcity."""
        # Register processes
        self.orchestrator.register_process(self.reasoning_process)
        self.orchestrator.register_process(self.learning_process)
        self.orchestrator.register_process(self.perception_process)
        
        # Request more attention than available
        self.orchestrator.request_attention("test-reasoning", 50.0, AttentionPriority.HIGH)
        self.orchestrator.request_attention("test-learning", 40.0, AttentionPriority.NORMAL)
        self.orchestrator.request_attention("test-perception", 30.0, AttentionPriority.LOW)
        
        # Allocate attention
        self.orchestrator.allocate_attention()
        
        # Verify high priority gets full allocation
        self.assertEqual(self.orchestrator.processes["test-reasoning"].attention_allocated, 50.0)
        
        # Verify normal priority gets full allocation
        self.assertEqual(self.orchestrator.processes["test-learning"].attention_allocated, 40.0)
        
        # Verify low priority gets remaining
        self.assertEqual(self.orchestrator.processes["test-perception"].attention_allocated, 10.0)
    
    def test_pattern_submission(self):
        """Test pattern submission and storage."""
        # Register process
        self.orchestrator.register_process(self.perception_process)
        
        # Create and submit pattern
        pattern = Pattern(
            pattern_id="test-pattern-001",
            source_process="test-perception",
            pattern_type="visual",
            content={"type": "edge", "confidence": 0.9},
            confidence=0.9,
            attention_value=0.8
        )
        
        self.orchestrator.submit_pattern(pattern)
        
        # Verify pattern is in knowledge base
        self.assertIn("test-pattern-001", self.orchestrator.knowledge_base.patterns)
        
        # Verify process metrics updated
        self.assertEqual(self.perception_process.patterns_discovered, 1)
    
    def test_pattern_propagation(self):
        """Test pattern propagation across processes."""
        # Register processes
        self.orchestrator.register_process(self.perception_process)
        self.orchestrator.register_process(self.reasoning_process)
        self.orchestrator.register_process(self.learning_process)
        
        # Submit pattern from perception
        pattern = Pattern(
            pattern_id="test-pattern-002",
            source_process="test-perception",
            pattern_type="visual",
            content={"type": "object", "confidence": 0.85},
            confidence=0.85,
            attention_value=0.7
        )
        
        self.orchestrator.submit_pattern(pattern)
        
        # Propagate patterns
        self.orchestrator.propagate_patterns()
        
        # Verify pattern was propagated to other processes
        # (not to source process)
        self.assertGreater(self.reasoning_process.patterns_consumed, 0)
        self.assertGreater(self.learning_process.patterns_consumed, 0)
        self.assertEqual(self.perception_process.patterns_consumed, 0)
        
        # Verify synergy metrics updated
        self.assertGreater(self.orchestrator.metrics.cross_module_patterns, 0)
    
    def test_bottleneck_detection(self):
        """Test bottleneck detection for stuck processes."""
        # Register process
        self.orchestrator.register_process(self.reasoning_process)
        
        # Set process to active but with old activity timestamp
        self.reasoning_process.state = ProcessState.ACTIVE
        self.reasoning_process.last_activity = datetime.now() - timedelta(seconds=60)
        
        # Detect bottlenecks
        bottlenecks = self.orchestrator.detect_bottlenecks()
        
        # Verify bottleneck detected
        self.assertIn("test-reasoning", bottlenecks)
        self.assertEqual(self.reasoning_process.state, ProcessState.STUCK)
        self.assertEqual(self.reasoning_process.bottleneck_count, 1)
    
    def test_bottleneck_resolution(self):
        """Test bottleneck resolution mechanism."""
        # Register process
        self.orchestrator.register_process(self.reasoning_process)
        self.reasoning_process.state = ProcessState.STUCK
        self.reasoning_process.attention_requested = 20.0
        
        # Add some patterns to knowledge base
        pattern = Pattern(
            pattern_id="helper-pattern",
            source_process="external",
            pattern_type="reasoning",
            content={"hint": "solution"},
            confidence=0.9,
            attention_value=0.9
        )
        self.orchestrator.knowledge_base.add_pattern(pattern)
        
        # Resolve bottleneck
        self.orchestrator.resolve_bottlenecks(["test-reasoning"])
        
        # Verify resolution attempted
        self.assertGreater(self.orchestrator.metrics.bottlenecks_resolved, 0)
        
        # Verify patterns provided to stuck process
        self.assertGreater(self.reasoning_process.patterns_consumed, 0)
    
    def test_emergent_behavior_detection(self):
        """Test detection of emergent behaviors."""
        # Register processes
        self.orchestrator.register_process(self.perception_process)
        self.orchestrator.register_process(self.reasoning_process)
        self.orchestrator.register_process(self.learning_process)
        
        # Create high-attention pattern consumed by multiple processes
        pattern = Pattern(
            pattern_id="emergent-pattern",
            source_process="test-perception",
            pattern_type="complex",
            content={"type": "emergent", "complexity": 0.95},
            confidence=0.95,
            attention_value=0.9
        )
        
        # Manually add consumers to simulate propagation
        pattern.consumers = {"test-reasoning", "test-learning", "other-process"}
        
        self.orchestrator.knowledge_base.add_pattern(pattern)
        
        # Detect emergent behaviors
        self.orchestrator.detect_emergent_behaviors()
        
        # Verify emergent behavior detected
        self.assertGreater(len(self.orchestrator.emergent_behaviors), 0)
        self.assertGreater(self.orchestrator.metrics.emergent_behaviors, 0)
    
    def test_orchestration_loop(self):
        """Test the main orchestration loop."""
        # Register processes
        self.orchestrator.register_process(self.reasoning_process)
        self.orchestrator.register_process(self.learning_process)
        
        # Start orchestrator
        self.orchestrator.start()
        self.assertTrue(self.orchestrator.running)
        
        # Request attention
        self.orchestrator.request_attention("test-reasoning", 30.0, AttentionPriority.HIGH)
        
        # Submit pattern
        pattern = Pattern(
            pattern_id="loop-test-pattern",
            source_process="test-reasoning",
            pattern_type="inference",
            content={"conclusion": "test"},
            confidence=0.8,
            attention_value=0.6
        )
        self.orchestrator.submit_pattern(pattern)
        
        # Let loop run
        time.sleep(1)
        
        # Verify orchestrator is functioning
        self.assertGreater(self.orchestrator.processes["test-reasoning"].attention_allocated, 0)
        
        # Stop orchestrator
        self.orchestrator.stop()
        self.assertFalse(self.orchestrator.running)
    
    def test_synergy_metrics(self):
        """Test synergy metrics collection."""
        # Register processes
        self.orchestrator.register_process(self.reasoning_process)
        self.orchestrator.register_process(self.learning_process)
        
        # Set processes to active
        self.reasoning_process.state = ProcessState.ACTIVE
        self.learning_process.state = ProcessState.ACTIVE
        
        # Request and allocate attention
        self.orchestrator.request_attention("test-reasoning", 40.0, AttentionPriority.HIGH)
        self.orchestrator.request_attention("test-learning", 30.0, AttentionPriority.NORMAL)
        self.orchestrator.allocate_attention()
        
        # Submit and propagate patterns
        pattern = Pattern(
            pattern_id="metrics-pattern",
            source_process="test-reasoning",
            pattern_type="inference",
            content={"data": "test"},
            confidence=0.85,
            attention_value=0.7
        )
        self.orchestrator.submit_pattern(pattern)
        self.orchestrator.propagate_patterns()
        
        # Get metrics
        metrics = self.orchestrator.get_metrics()
        
        # Verify metrics
        self.assertIsInstance(metrics, SynergyMetrics)
        self.assertEqual(metrics.active_processes, 2)
        self.assertGreater(metrics.attention_efficiency, 0)
        self.assertGreater(metrics.total_patterns_shared, 0)
    
    def test_status_export(self):
        """Test status retrieval and export."""
        # Register process
        self.orchestrator.register_process(self.reasoning_process)
        
        # Get status
        status = self.orchestrator.get_status()
        
        # Verify status structure
        self.assertIn("running", status)
        self.assertIn("total_processes", status)
        self.assertIn("metrics", status)
        self.assertIn("processes", status)
        
        self.assertEqual(status["total_processes"], 1)
        self.assertIn("test-reasoning", status["processes"])
    
    def test_knowledge_base_operations(self):
        """Test hypergraph knowledge base operations."""
        kb = self.orchestrator.knowledge_base
        
        # Add patterns of different types
        pattern1 = Pattern(
            pattern_id="kb-pattern-1",
            source_process="test",
            pattern_type="visual",
            content={},
            confidence=0.8,
            attention_value=0.6
        )
        
        pattern2 = Pattern(
            pattern_id="kb-pattern-2",
            source_process="test",
            pattern_type="visual",
            content={},
            confidence=0.9,
            attention_value=0.8
        )
        
        pattern3 = Pattern(
            pattern_id="kb-pattern-3",
            source_process="test",
            pattern_type="reasoning",
            content={},
            confidence=0.7,
            attention_value=0.5
        )
        
        kb.add_pattern(pattern1)
        kb.add_pattern(pattern2)
        kb.add_pattern(pattern3)
        
        # Test retrieval by type
        visual_patterns = kb.get_patterns_by_type("visual")
        self.assertEqual(len(visual_patterns), 2)
        
        reasoning_patterns = kb.get_patterns_by_type("reasoning")
        self.assertEqual(len(reasoning_patterns), 1)
        
        # Test high attention retrieval
        high_attention = kb.get_high_attention_patterns(threshold=0.7)
        self.assertEqual(len(high_attention), 1)
        self.assertEqual(high_attention[0].pattern_id, "kb-pattern-2")
        
        # Test attention propagation
        kb.propagate_attention("kb-pattern-3", 0.3)
        self.assertEqual(kb.patterns["kb-pattern-3"].attention_value, 0.8)


class TestCognitiveProcess(unittest.TestCase):
    """Test cases for CognitiveProcess."""
    
    def test_process_creation(self):
        """Test cognitive process creation."""
        process = CognitiveProcess(
            process_id="test-process",
            process_type="test",
            priority=AttentionPriority.HIGH
        )
        
        self.assertEqual(process.process_id, "test-process")
        self.assertEqual(process.process_type, "test")
        self.assertEqual(process.state, ProcessState.IDLE)
        self.assertEqual(process.priority, AttentionPriority.HIGH)
    
    def test_metrics_update(self):
        """Test process metrics updating."""
        process = CognitiveProcess(
            process_id="test-process",
            process_type="test"
        )
        
        initial_time = process.last_activity
        time.sleep(0.1)
        
        process.update_metrics("accuracy", 0.95)
        
        self.assertEqual(process.performance_metrics["accuracy"], 0.95)
        self.assertGreater(process.last_activity, initial_time)


class TestPattern(unittest.TestCase):
    """Test cases for Pattern."""
    
    def test_pattern_creation(self):
        """Test pattern creation."""
        pattern = Pattern(
            pattern_id="test-pattern",
            source_process="test-source",
            pattern_type="test-type",
            content={"key": "value"},
            confidence=0.9,
            attention_value=0.8
        )
        
        self.assertEqual(pattern.pattern_id, "test-pattern")
        self.assertEqual(pattern.source_process, "test-source")
        self.assertEqual(pattern.pattern_type, "test-type")
        self.assertEqual(pattern.confidence, 0.9)
        self.assertEqual(pattern.attention_value, 0.8)
        self.assertEqual(len(pattern.consumers), 0)
    
    def test_pattern_hashable(self):
        """Test that patterns are hashable."""
        pattern = Pattern(
            pattern_id="hashable-pattern",
            source_process="test",
            pattern_type="test",
            content={},
            confidence=0.8,
            attention_value=0.7
        )
        
        # Should be able to add to set
        pattern_set = {pattern}
        self.assertEqual(len(pattern_set), 1)


class TestSynergyMetrics(unittest.TestCase):
    """Test cases for SynergyMetrics."""
    
    def test_metrics_creation(self):
        """Test metrics creation with defaults."""
        metrics = SynergyMetrics()
        
        self.assertEqual(metrics.cross_module_patterns, 0)
        self.assertEqual(metrics.bottlenecks_resolved, 0)
        self.assertEqual(metrics.emergent_behaviors, 0)
        self.assertEqual(metrics.attention_efficiency, 0.0)
    
    def test_metrics_to_dict(self):
        """Test metrics conversion to dictionary."""
        metrics = SynergyMetrics(
            cross_module_patterns=10,
            bottlenecks_resolved=2,
            emergent_behaviors=1,
            attention_efficiency=0.85
        )
        
        metrics_dict = metrics.to_dict()
        
        self.assertEqual(metrics_dict["cross_module_patterns"], 10)
        self.assertEqual(metrics_dict["bottlenecks_resolved"], 2)
        self.assertEqual(metrics_dict["emergent_behaviors"], 1)
        self.assertEqual(metrics_dict["attention_efficiency"], 0.85)
        self.assertIn("timestamp", metrics_dict)


if __name__ == "__main__":
    unittest.main()

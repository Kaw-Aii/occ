"""
Cognitive Integration Tests for OpenCog Collection
==================================================

Test suite for verifying cognitive synergy between components:
- AAR (Agent-Arena-Relation) self-awareness
- Hypergraph dynamics
- Cognitive synergy orchestrator
- Pattern recognition and sharing
- Meta-cognitive reasoning

Author: OpenCog Collection Contributors
License: GPL-3.0+
"""

import sys
import os
import unittest
import numpy as np
from datetime import datetime

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from self_awareness_aar import AARCore, AgentState, ArenaState, RelationState
    from hypergraph_dynamics import HypergraphDynamics, HypergraphSynergyBridge
    from cognitive_synergy_orchestrator import CognitiveSynergyOrchestrator, CognitiveTask
    MODULES_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Some modules not available: {e}")
    MODULES_AVAILABLE = False


class TestAARCore(unittest.TestCase):
    """Test AAR (Agent-Arena-Relation) core functionality."""
    
    @unittest.skipIf(not MODULES_AVAILABLE, "Required modules not available")
    def setUp(self):
        """Set up test fixtures."""
        self.aar = AARCore()
    
    @unittest.skipIf(not MODULES_AVAILABLE, "Required modules not available")
    def test_aar_initialization(self):
        """Test AAR core initialization."""
        self.assertIsNotNone(self.aar.agent)
        self.assertIsNotNone(self.aar.arena)
        self.assertIsNotNone(self.aar.relation)
        self.assertEqual(self.aar.meta_cognitive_depth, 0)
    
    @unittest.skipIf(not MODULES_AVAILABLE, "Required modules not available")
    def test_agent_update(self):
        """Test agent state updates."""
        process_activations = {
            'reasoning': 0.8,
            'perception': 0.6,
            'learning': 0.7
        }
        
        self.aar.update_agent(process_activations, new_goal="test_goal")
        
        self.assertGreater(self.aar.agent.action_potential, 0)
        self.assertIn("test_goal", self.aar.agent.goal_stack)
        self.assertEqual(len(self.aar.agent.process_activations), 3)
    
    @unittest.skipIf(not MODULES_AVAILABLE, "Required modules not available")
    def test_arena_update(self):
        """Test arena state updates."""
        self.aar.update_arena(
            atom_count=100,
            link_count=150,
            pattern_matches=80,
            total_patterns=100
        )
        
        self.assertEqual(self.aar.arena.knowledge_density, 1.5)
        self.assertEqual(self.aar.arena.coherence_measure, 0.8)
        self.assertGreater(len(self.aar.arena.state_history), 0)
    
    @unittest.skipIf(not MODULES_AVAILABLE, "Required modules not available")
    def test_relation_update(self):
        """Test relation state updates."""
        # Set up agent and arena
        self.aar.update_agent({'process_a': 0.7})
        self.aar.update_arena(50, 75, 40, 50)
        
        # Update relation
        self.aar.update_relation()
        
        self.assertGreater(self.aar.relation.self_coherence, 0)
        self.assertIsNotNone(self.aar.relation.identity_vector)
        self.assertIn('identity_norm', self.aar.relation.self_model)
    
    @unittest.skipIf(not MODULES_AVAILABLE, "Required modules not available")
    def test_meta_cognitive_step(self):
        """Test meta-cognitive reasoning."""
        # Set up state
        self.aar.update_agent({'reasoning': 0.8})
        self.aar.update_arena(100, 150, 80, 100)
        self.aar.update_relation()
        
        # Perform meta-cognitive step
        assessment = self.aar.meta_cognitive_step()
        
        self.assertIn('timestamp', assessment)
        self.assertIn('self_state', assessment)
        self.assertIn('insights', assessment)
        self.assertIn('recommendations', assessment)
        self.assertEqual(self.aar.meta_cognitive_depth, 1)
    
    @unittest.skipIf(not MODULES_AVAILABLE, "Required modules not available")
    def test_self_perception(self):
        """Test self-perception functionality."""
        perception = self.aar.perceive_self()
        
        self.assertIn('agent', perception)
        self.assertIn('arena', perception)
        self.assertIn('relation', perception)
        self.assertIn('meta', perception)


class TestHypergraphDynamics(unittest.TestCase):
    """Test hypergraph dynamics functionality."""
    
    @unittest.skipIf(not MODULES_AVAILABLE, "Required modules not available")
    def setUp(self):
        """Set up test fixtures."""
        self.hg = HypergraphDynamics()
    
    @unittest.skipIf(not MODULES_AVAILABLE, "Required modules not available")
    def test_hypergraph_initialization(self):
        """Test hypergraph initialization."""
        self.assertEqual(len(self.hg.nodes), 0)
        self.assertEqual(len(self.hg.edges), 0)
        self.assertIsNotNone(self.hg.metrics)
    
    @unittest.skipIf(not MODULES_AVAILABLE, "Required modules not available")
    def test_add_nodes(self):
        """Test adding nodes to hypergraph."""
        node = self.hg.add_node("concept_1", "concept")
        
        self.assertEqual(node.node_id, "concept_1")
        self.assertEqual(node.node_type, "concept")
        self.assertIn("concept_1", self.hg.nodes)
        self.assertEqual(self.hg.metrics['total_nodes'], 1)
    
    @unittest.skipIf(not MODULES_AVAILABLE, "Required modules not available")
    def test_add_edges(self):
        """Test adding hyperedges."""
        self.hg.add_node("node_1", "concept")
        self.hg.add_node("node_2", "concept")
        self.hg.add_node("node_3", "concept")
        
        edge = self.hg.add_edge(
            "edge_1",
            {"node_1", "node_2", "node_3"},
            "relation"
        )
        
        self.assertEqual(edge.edge_id, "edge_1")
        self.assertEqual(len(edge.nodes), 3)
        self.assertIn("edge_1", self.hg.edges)
        self.assertEqual(self.hg.metrics['total_edges'], 1)
    
    @unittest.skipIf(not MODULES_AVAILABLE, "Required modules not available")
    def test_attention_spreading(self):
        """Test attention spreading mechanism."""
        # Create small network
        for i in range(5):
            self.hg.add_node(f"node_{i}", "concept")
        
        self.hg.add_edge("edge_0", {"node_0", "node_1"}, "link")
        self.hg.add_edge("edge_1", {"node_1", "node_2"}, "link")
        self.hg.add_edge("edge_2", {"node_2", "node_3"}, "link")
        
        # Spread attention
        attention = self.hg.spread_attention(["node_0"], initial_attention=1.0)
        
        self.assertGreater(len(attention), 1)
        self.assertIn("node_0", attention)
        self.assertEqual(attention["node_0"], 1.0)
    
    @unittest.skipIf(not MODULES_AVAILABLE, "Required modules not available")
    def test_pattern_activation(self):
        """Test pattern activation and coherence."""
        # Create connected pattern
        for i in range(4):
            self.hg.add_node(f"pattern_node_{i}", "concept")
        
        self.hg.add_edge("p_edge_0", {"pattern_node_0", "pattern_node_1"}, "link")
        self.hg.add_edge("p_edge_1", {"pattern_node_1", "pattern_node_2"}, "link")
        
        # Activate pattern
        coherence = self.hg.activate_pattern(
            ["pattern_node_0", "pattern_node_1", "pattern_node_2"]
        )
        
        self.assertGreaterEqual(coherence, 0.0)
        self.assertLessEqual(coherence, 1.0)
    
    @unittest.skipIf(not MODULES_AVAILABLE, "Required modules not available")
    def test_structural_metrics(self):
        """Test structural metrics computation."""
        # Create network
        for i in range(10):
            self.hg.add_node(f"metric_node_{i}", "concept")
        
        for i in range(5):
            self.hg.add_edge(
                f"metric_edge_{i}",
                {f"metric_node_{i}", f"metric_node_{i+1}"},
                "link"
            )
        
        metrics = self.hg.compute_structural_metrics()
        
        self.assertIn('total_nodes', metrics)
        self.assertIn('total_edges', metrics)
        self.assertIn('average_degree', metrics)
        self.assertIn('attention_entropy', metrics)
        self.assertEqual(metrics['total_nodes'], 10)
        self.assertEqual(metrics['total_edges'], 5)
    
    @unittest.skipIf(not MODULES_AVAILABLE, "Required modules not available")
    def test_attention_decay(self):
        """Test attention decay mechanism."""
        node = self.hg.add_node("decay_node", "concept")
        node.attention = 1.0
        
        self.hg.decay_attention(decay_rate=0.5)
        
        self.assertEqual(node.attention, 0.5)
    
    @unittest.skipIf(not MODULES_AVAILABLE, "Required modules not available")
    def test_state_export_import(self):
        """Test state export and import."""
        # Create state
        self.hg.add_node("export_node", "concept")
        self.hg.add_edge("export_edge", {"export_node"}, "self_link")
        
        # Export
        state = self.hg.export_state()
        
        self.assertIn('nodes', state)
        self.assertIn('edges', state)
        self.assertIn('metrics', state)
        
        # Import to new hypergraph
        hg2 = HypergraphDynamics()
        hg2.import_state(state)
        
        self.assertEqual(len(hg2.nodes), len(self.hg.nodes))
        self.assertEqual(len(hg2.edges), len(self.hg.edges))


class TestHypergraphSynergyBridge(unittest.TestCase):
    """Test hypergraph synergy bridge functionality."""
    
    @unittest.skipIf(not MODULES_AVAILABLE, "Required modules not available")
    def setUp(self):
        """Set up test fixtures."""
        self.hg = HypergraphDynamics()
        self.bridge = HypergraphSynergyBridge(self.hg)
    
    @unittest.skipIf(not MODULES_AVAILABLE, "Required modules not available")
    def test_bridge_initialization(self):
        """Test bridge initialization."""
        self.assertIsNotNone(self.bridge.hypergraph)
        self.assertIsInstance(self.bridge.synergy_metrics, dict)
    
    @unittest.skipIf(not MODULES_AVAILABLE, "Required modules not available")
    def test_aar_integration(self):
        """Test AAR state integration."""
        agent_activations = {
            'process_a': 0.8,
            'process_b': 0.6,
            'process_c': 0.7
        }
        
        arena_knowledge = {
            'atom_count': 100,
            'link_count': 150
        }
        
        metrics = self.bridge.integrate_aar_state(agent_activations, arena_knowledge)
        
        self.assertIn('agent_coverage', metrics)
        self.assertIn('average_attention', metrics)
    
    @unittest.skipIf(not MODULES_AVAILABLE, "Required modules not available")
    def test_pattern_extraction(self):
        """Test pattern extraction for synergy."""
        # Create connected patterns
        for i in range(10):
            node = self.hg.add_node(f"synergy_node_{i}", "concept")
            node.attention = 0.8 if i < 5 else 0.3
        
        for i in range(4):
            self.hg.add_edge(
                f"synergy_edge_{i}",
                {f"synergy_node_{i}", f"synergy_node_{i+1}"},
                "link"
            )
        
        patterns = self.bridge.extract_patterns_for_synergy(min_coherence=0.3)
        
        self.assertIsInstance(patterns, list)


class TestCognitiveSynergyOrchestrator(unittest.TestCase):
    """Test cognitive synergy orchestrator functionality."""
    
    @unittest.skipIf(not MODULES_AVAILABLE, "Required modules not available")
    def setUp(self):
        """Set up test fixtures."""
        self.orchestrator = CognitiveSynergyOrchestrator(
            enable_aar=True,
            enable_hypergraph=True
        )
    
    @unittest.skipIf(not MODULES_AVAILABLE, "Required modules not available")
    def test_orchestrator_initialization(self):
        """Test orchestrator initialization."""
        self.assertIsNotNone(self.orchestrator.components)
        self.assertIsNotNone(self.orchestrator.metrics)
        self.assertIsInstance(self.orchestrator.shared_patterns, dict)
    
    @unittest.skipIf(not MODULES_AVAILABLE, "Required modules not available")
    def test_task_submission(self):
        """Test task submission."""
        task = CognitiveTask(
            task_id="test_task_1",
            task_type="pattern_recognition",
            priority=0.8,
            data={'source_nodes': ['node_1']},
            required_components=['hypergraph']
        )
        
        task_id = self.orchestrator.submit_task(task)
        
        self.assertEqual(task_id, "test_task_1")
        self.assertIn(task_id, self.orchestrator.active_tasks)
    
    @unittest.skipIf(not MODULES_AVAILABLE, "Required modules not available")
    def test_pattern_recognition_task(self):
        """Test pattern recognition task processing."""
        # Set up hypergraph
        if self.orchestrator.hypergraph:
            for i in range(5):
                self.orchestrator.hypergraph.add_node(f"test_node_{i}", "concept")
        
        task = CognitiveTask(
            task_id="pattern_task",
            task_type="pattern_recognition",
            priority=0.8,
            data={'source_nodes': ['test_node_0']},
            required_components=['hypergraph']
        )
        
        result = self.orchestrator.process_task(task)
        
        self.assertIn('attention_spread', result)
        self.assertIn('patterns_found', result)
    
    @unittest.skipIf(not MODULES_AVAILABLE, "Required modules not available")
    def test_self_awareness_task(self):
        """Test self-awareness task processing."""
        task = CognitiveTask(
            task_id="awareness_task",
            task_type="self_awareness",
            priority=0.9,
            data={},
            required_components=['aar']
        )
        
        result = self.orchestrator.process_task(task)
        
        self.assertIn('timestamp', result)
        self.assertIn('depth', result)
    
    @unittest.skipIf(not MODULES_AVAILABLE, "Required modules not available")
    def test_bottleneck_detection(self):
        """Test bottleneck detection."""
        bottlenecks = self.orchestrator.detect_bottlenecks()
        
        self.assertIsInstance(bottlenecks, list)
    
    @unittest.skipIf(not MODULES_AVAILABLE, "Required modules not available")
    def test_metrics_update(self):
        """Test metrics update."""
        self.orchestrator.update_metrics()
        
        self.assertGreaterEqual(self.orchestrator.metrics.synergy_index, 0.0)
        self.assertLessEqual(self.orchestrator.metrics.synergy_index, 1.0)
    
    @unittest.skipIf(not MODULES_AVAILABLE, "Required modules not available")
    def test_status_report(self):
        """Test status reporting."""
        status = self.orchestrator.get_status()
        
        self.assertIn('timestamp', status)
        self.assertIn('components', status)
        self.assertIn('tasks', status)
        self.assertIn('metrics', status)
        self.assertIn('patterns', status)
    
    @unittest.skipIf(not MODULES_AVAILABLE, "Required modules not available")
    def test_state_export(self):
        """Test state export."""
        state = self.orchestrator.export_state()
        
        self.assertIn('timestamp', state)
        self.assertIn('status', state)
        self.assertIn('metrics_history', state)


class TestCognitiveSynergyIntegration(unittest.TestCase):
    """Integration tests for cognitive synergy across all components."""
    
    @unittest.skipIf(not MODULES_AVAILABLE, "Required modules not available")
    def test_full_cognitive_cycle(self):
        """Test complete cognitive cycle with all components."""
        # Initialize orchestrator
        orchestrator = CognitiveSynergyOrchestrator(
            enable_aar=True,
            enable_hypergraph=True
        )
        
        # Phase 1: Knowledge integration
        knowledge_task = CognitiveTask(
            task_id="integrate_knowledge",
            task_type="knowledge_integration",
            priority=0.9,
            data={
                'knowledge_items': [
                    {'id': 'concept_a', 'type': 'concept'},
                    {'id': 'concept_b', 'type': 'concept'},
                    {'id': 'concept_c', 'type': 'concept'}
                ]
            },
            required_components=['hypergraph', 'aar']
        )
        
        result1 = orchestrator.process_task(knowledge_task)
        self.assertIn('integrated_items', result1)
        
        # Phase 2: Pattern recognition
        pattern_task = CognitiveTask(
            task_id="recognize_patterns",
            task_type="pattern_recognition",
            priority=0.8,
            data={'source_nodes': ['concept_a']},
            required_components=['hypergraph']
        )
        
        result2 = orchestrator.process_task(pattern_task)
        self.assertIn('patterns_found', result2)
        
        # Phase 3: Self-awareness
        awareness_task = CognitiveTask(
            task_id="self_awareness",
            task_type="self_awareness",
            priority=0.9,
            data={},
            required_components=['aar']
        )
        
        result3 = orchestrator.process_task(awareness_task)
        self.assertIn('self_state', result3)
        
        # Phase 4: Synergy optimization
        optimize_task = CognitiveTask(
            task_id="optimize_synergy",
            task_type="synergy_optimization",
            priority=1.0,
            data={},
            required_components=['aar', 'hypergraph']
        )
        
        result4 = orchestrator.process_task(optimize_task)
        self.assertIn('new_synergy_index', result4)
        
        # Verify synergy metrics improved
        orchestrator.update_metrics()
        self.assertGreater(orchestrator.metrics.synergy_index, 0.0)


def run_tests():
    """Run all cognitive integration tests."""
    print("=" * 60)
    print("OpenCog Collection - Cognitive Integration Tests")
    print("=" * 60)
    print()
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test cases
    suite.addTests(loader.loadTestsFromTestCase(TestAARCore))
    suite.addTests(loader.loadTestsFromTestCase(TestHypergraphDynamics))
    suite.addTests(loader.loadTestsFromTestCase(TestHypergraphSynergyBridge))
    suite.addTests(loader.loadTestsFromTestCase(TestCognitiveSynergyOrchestrator))
    suite.addTests(loader.loadTestsFromTestCase(TestCognitiveSynergyIntegration))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print()
    print("=" * 60)
    print("Test Summary")
    print("=" * 60)
    print(f"Tests run: {result.testsRun}")
    print(f"Successes: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Skipped: {len(result.skipped)}")
    print()
    
    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_tests()
    sys.exit(0 if success else 1)

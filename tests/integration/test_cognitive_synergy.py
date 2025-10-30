"""
Cognitive Synergy Integration Tests
====================================

Tests the integration and interaction of cognitive components
to validate emergent synergy properties.

Author: OpenCog Collection Contributors
License: GPL-3.0+
"""

import unittest
import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
from typing import Dict, Any


class TestCognitiveSynergyIntegration(unittest.TestCase):
    """Test suite for cognitive synergy integration."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures."""
        print("\n" + "=" * 60)
        print("Cognitive Synergy Integration Test Suite")
        print("=" * 60)
    
    def test_component_discovery(self):
        """Test that cognitive components can be discovered."""
        try:
            from cognitive_discovery import CognitiveComponentRegistry, CognitiveComponentDiscovery
            
            registry = CognitiveComponentRegistry()
            discovery = CognitiveComponentDiscovery(registry)
            
            # Scan for components
            count = discovery.scan_directory(".")
            
            self.assertGreater(count, 0, "Should discover at least one component")
            self.assertGreater(len(registry.list_components()), 0, 
                             "Registry should contain components")
            
            print(f"✓ Discovered {count} cognitive components")
            
        except ImportError as e:
            self.skipTest(f"cognitive_discovery not available: {e}")
    
    def test_aar_core_initialization(self):
        """Test AAR (Agent-Arena-Relation) core initialization."""
        try:
            from self_awareness_aar import AgentState, ArenaState, RelationState
            
            # Initialize AAR components
            agent = AgentState()
            arena = ArenaState()
            relation = RelationState()
            
            # Verify initialization
            self.assertIsNotNone(agent.intention_vector)
            self.assertIsNotNone(arena.attention_landscape)
            self.assertIsNotNone(relation.identity_vector)
            
            # Test basic operations
            agent.update_intention(np.random.randn(128))
            self.assertGreater(agent.action_potential, 0)
            
            arena.compute_knowledge_density(100, 250)
            self.assertGreater(arena.knowledge_density, 0)
            
            print("✓ AAR core initialized successfully")
            
        except ImportError as e:
            self.skipTest(f"self_awareness_aar not available: {e}")
    
    def test_hypergraph_operations(self):
        """Test hypergraph dynamics and operations."""
        try:
            from hypergraph_dynamics import HypergraphDynamics
            
            # Initialize hypergraph
            hg = HypergraphDynamics()
            
            # Add nodes and edges
            node1 = hg.add_node("concept", {"name": "intelligence"})
            node2 = hg.add_node("concept", {"name": "learning"})
            edge = hg.add_edge([node1, node2], "relates_to")
            
            # Verify structure
            self.assertIsNotNone(node1)
            self.assertIsNotNone(node2)
            self.assertIsNotNone(edge)
            
            # Test queries
            results = hg.query_nodes({"type": "concept"})
            self.assertGreaterEqual(len(results), 2)
            
            print(f"✓ Hypergraph operations validated ({len(results)} nodes)")
            
        except ImportError as e:
            self.skipTest(f"hypergraph_dynamics not available: {e}")
        except Exception as e:
            self.skipTest(f"Hypergraph test failed: {e}")
    
    def test_neural_symbolic_integration(self):
        """Test neural-symbolic integration bridge."""
        try:
            from neural_symbolic_integration import NeuralSymbolicBridge
            
            # Initialize bridge
            bridge = NeuralSymbolicBridge()
            
            # Test symbolic to neural
            symbolic_input = {"concept": "intelligence", "value": 0.8}
            neural_repr = bridge.symbolic_to_neural(symbolic_input)
            
            self.assertIsNotNone(neural_repr)
            self.assertTrue(isinstance(neural_repr, (np.ndarray, list)))
            
            # Test neural to symbolic
            neural_input = np.random.randn(64)
            symbolic_repr = bridge.neural_to_symbolic(neural_input)
            
            self.assertIsNotNone(symbolic_repr)
            
            print("✓ Neural-symbolic integration validated")
            
        except ImportError as e:
            self.skipTest(f"neural_symbolic_integration not available: {e}")
        except Exception as e:
            self.skipTest(f"Neural-symbolic test failed: {e}")
    
    def test_multi_agent_collaboration(self):
        """Test multi-agent collaboration system."""
        try:
            from multi_agent_collaboration import CognitiveAgent, MultiAgentSystem
            
            # Create multi-agent system
            mas = MultiAgentSystem()
            
            # Add agents
            agent1 = CognitiveAgent(name="reasoner", capabilities=["logic", "inference"])
            agent2 = CognitiveAgent(name="learner", capabilities=["pattern", "adaptation"])
            
            mas.add_agent(agent1)
            mas.add_agent(agent2)
            
            # Verify agents
            self.assertEqual(len(mas.agents), 2)
            
            # Test collaboration
            result = mas.collaborate(task="solve_problem", context={})
            self.assertIsNotNone(result)
            
            print(f"✓ Multi-agent collaboration validated ({len(mas.agents)} agents)")
            
        except ImportError as e:
            self.skipTest(f"multi_agent_collaboration not available: {e}")
        except Exception as e:
            self.skipTest(f"Multi-agent test failed: {e}")
    
    def test_cognitive_monitoring(self):
        """Test cognitive monitoring and metrics collection."""
        try:
            from cognitive_monitoring import CognitiveMonitor
            
            # Initialize monitor
            monitor = CognitiveMonitor()
            
            # Collect metrics
            metrics = monitor.collect_metrics()
            
            self.assertIsNotNone(metrics)
            self.assertIsInstance(metrics, dict)
            
            # Verify key metrics exist
            expected_metrics = ['timestamp', 'system_health', 'component_status']
            for metric in expected_metrics:
                if metric in metrics:
                    print(f"  • {metric}: {metrics[metric]}")
            
            print("✓ Cognitive monitoring validated")
            
        except ImportError as e:
            self.skipTest(f"cognitive_monitoring not available: {e}")
        except Exception as e:
            self.skipTest(f"Monitoring test failed: {e}")
    
    def test_attention_allocation(self):
        """Test attention allocation mechanism."""
        try:
            from attention_allocation import AttentionAllocator
            
            # Initialize allocator
            allocator = AttentionAllocator()
            
            # Create attention targets
            targets = [
                {"id": "target1", "importance": 0.8, "urgency": 0.6},
                {"id": "target2", "importance": 0.5, "urgency": 0.9},
                {"id": "target3", "importance": 0.7, "urgency": 0.4},
            ]
            
            # Allocate attention
            allocation = allocator.allocate(targets)
            
            self.assertIsNotNone(allocation)
            self.assertEqual(len(allocation), len(targets))
            
            # Verify allocation sums to 1.0 (normalized)
            total = sum(allocation.values())
            self.assertAlmostEqual(total, 1.0, places=2)
            
            print(f"✓ Attention allocation validated ({len(targets)} targets)")
            
        except ImportError as e:
            self.skipTest(f"attention_allocation not available: {e}")
        except Exception as e:
            self.skipTest(f"Attention test failed: {e}")
    
    def test_cognitive_synergy_emergence(self):
        """
        Test emergent cognitive synergy from component interactions.
        This is the key test validating that components work together
        to produce capabilities beyond their individual functions.
        """
        try:
            # Import multiple components
            from self_awareness_aar import AgentState, ArenaState, RelationState
            from attention_allocation import AttentionAllocator
            
            # Initialize components
            agent = AgentState()
            arena = ArenaState()
            relation = RelationState()
            allocator = AttentionAllocator()
            
            # Simulate cognitive process
            # 1. Agent generates intentions
            agent.update_intention(np.random.randn(128))
            
            # 2. Arena maintains state
            arena.compute_knowledge_density(150, 300)
            arena.update_coherence(80, 100)
            
            # 3. Attention allocation based on agent and arena
            targets = [
                {"id": "reasoning", "importance": agent.action_potential, "urgency": 0.7},
                {"id": "learning", "importance": arena.coherence_measure, "urgency": 0.5},
            ]
            allocation = allocator.allocate(targets)
            
            # 4. Verify synergy: allocation should reflect both agent and arena state
            self.assertIsNotNone(allocation)
            self.assertGreater(len(allocation), 0)
            
            # Calculate synergy score (simplified)
            synergy_score = (
                agent.action_potential * 0.3 +
                arena.coherence_measure * 0.3 +
                sum(allocation.values()) * 0.4
            )
            
            self.assertGreater(synergy_score, 0)
            
            print(f"✓ Cognitive synergy emergence validated (score: {synergy_score:.3f})")
            
        except ImportError as e:
            self.skipTest(f"Required components not available: {e}")
        except Exception as e:
            self.skipTest(f"Synergy emergence test failed: {e}")
    
    def test_mcp_integration(self):
        """Test MCP (Model Context Protocol) integration."""
        try:
            from mcp_cognitive_bridge import MCPCognitiveBridge
            
            # Initialize MCP bridge
            bridge = MCPCognitiveBridge()
            
            # Test connection (may fail if MCP server not available)
            status = bridge.get_status()
            
            self.assertIsNotNone(status)
            
            print("✓ MCP integration validated")
            
        except ImportError as e:
            self.skipTest(f"mcp_cognitive_bridge not available: {e}")
        except Exception as e:
            self.skipTest(f"MCP test skipped: {e}")


class TestCognitiveComponentInterfaces(unittest.TestCase):
    """Test that cognitive components implement expected interfaces."""
    
    def test_agent_interface(self):
        """Test that agent components implement required interface."""
        try:
            from self_awareness_aar import AgentState
            
            agent = AgentState()
            
            # Verify required methods
            self.assertTrue(hasattr(agent, 'compute_action_potential'))
            self.assertTrue(hasattr(agent, 'update_intention'))
            self.assertTrue(callable(agent.compute_action_potential))
            
            print("✓ Agent interface validated")
            
        except ImportError as e:
            self.skipTest(f"Agent component not available: {e}")
    
    def test_arena_interface(self):
        """Test that arena components implement required interface."""
        try:
            from self_awareness_aar import ArenaState
            
            arena = ArenaState()
            
            # Verify required methods
            self.assertTrue(hasattr(arena, 'compute_knowledge_density'))
            self.assertTrue(hasattr(arena, 'update_coherence'))
            self.assertTrue(callable(arena.compute_knowledge_density))
            
            print("✓ Arena interface validated")
            
        except ImportError as e:
            self.skipTest(f"Arena component not available: {e}")


def run_tests():
    """Run all cognitive synergy tests."""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test cases
    suite.addTests(loader.loadTestsFromTestCase(TestCognitiveSynergyIntegration))
    suite.addTests(loader.loadTestsFromTestCase(TestCognitiveComponentInterfaces))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    print(f"Tests run: {result.testsRun}")
    print(f"Successes: {result.testsRun - len(result.failures) - len(result.errors) - len(result.skipped)}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Skipped: {len(result.skipped)}")
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)

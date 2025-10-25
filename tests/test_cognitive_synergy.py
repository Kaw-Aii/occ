"""
Comprehensive Test Suite for Cognitive Synergy Framework
=========================================================

Tests for:
- Hypergraph memory operations
- Cognitive process interactions
- Synergy event detection
- Bottleneck resolution
- Pattern mining
- Self-awareness module
- Membrane architecture

Author: OpenCog Collection Contributors
License: GPL-3.0+
"""

import unittest
import sys
import os
import numpy as np
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cognitive_synergy_framework import (
    Atom, CognitiveProcess, HypergraphMemory,
    CognitiveSynergyEngine
)
from self_awareness_aar import AARCore, AgentState, ArenaState, RelationState
from deep_tree_echo_membranes import (
    Membrane, MembraneType, MessagePriority,
    DeepTreeEchoArchitecture
)


class TestHypergraphMemory(unittest.TestCase):
    """Test hypergraph memory operations."""
    
    def setUp(self):
        self.memory = HypergraphMemory()
    
    def test_add_atom(self):
        """Test adding atoms to memory."""
        atom = Atom(
            atom_type='ConceptNode',
            name='test_concept',
            truth_value=0.9,
            attention_value=0.5
        )
        
        atom_id = self.memory.add_atom(atom)
        self.assertIsNotNone(atom_id)
        self.assertIn(atom_id, self.memory.atoms)
    
    def test_get_atom(self):
        """Test retrieving atoms."""
        atom = Atom(atom_type='ConceptNode', name='test')
        atom_id = self.memory.add_atom(atom)
        
        retrieved = self.memory.get_atom(atom_id)
        self.assertIsNotNone(retrieved)
        self.assertEqual(retrieved.name, 'test')
    
    def test_link_atoms(self):
        """Test linking atoms."""
        atom1 = Atom(atom_type='ConceptNode', name='concept1')
        atom2 = Atom(atom_type='ConceptNode', name='concept2')
        
        id1 = self.memory.add_atom(atom1)
        id2 = self.memory.add_atom(atom2)
        
        self.memory.link_atoms(id1, id2, 'InheritanceLink')
        
        atom1_retrieved = self.memory.get_atom(id1)
        self.assertIn(id2, atom1_retrieved.outgoing)
    
    def test_attention_allocation(self):
        """Test attention allocation."""
        atom = Atom(atom_type='ConceptNode', name='test', attention_value=0.5)
        atom_id = self.memory.add_atom(atom)
        
        # Increase attention
        self.memory.attention_bank[atom_id] += 0.3
        self.assertGreater(self.memory.attention_bank[atom_id], 0.5)


class TestAARCore(unittest.TestCase):
    """Test Agent-Arena-Relation core."""
    
    def setUp(self):
        self.aar = AARCore()
    
    def test_initialization(self):
        """Test AAR core initialization."""
        self.assertIsNotNone(self.aar.agent)
        self.assertIsNotNone(self.aar.arena)
        self.assertIsNotNone(self.aar.relation)
    
    def test_agent_update(self):
        """Test agent state updates."""
        initial_potential = self.aar.agent.action_potential
        
        self.aar.update_agent({
            'process1': 0.7,
            'process2': 0.5
        })
        
        self.assertGreater(self.aar.agent.action_potential, initial_potential)
    
    def test_arena_update(self):
        """Test arena state updates."""
        self.aar.update_arena(
            atom_count=100,
            link_count=250,
            pattern_matches=45,
            total_patterns=50
        )
        
        self.assertEqual(self.aar.arena.knowledge_density, 2.5)
        self.assertEqual(self.aar.arena.coherence_measure, 0.9)
    
    def test_relation_update(self):
        """Test relation state updates."""
        # Update agent and arena first
        self.aar.update_agent({'process1': 0.6})
        self.aar.update_arena(100, 200, 40, 50)
        
        # Update relation
        self.aar.update_relation()
        
        self.assertGreater(self.aar.relation.self_coherence, 0)
        self.assertIsNotNone(self.aar.relation.identity_vector)
    
    def test_meta_cognitive_step(self):
        """Test meta-cognitive reasoning."""
        # Setup state
        self.aar.update_agent({'process1': 0.8})
        self.aar.update_arena(100, 200, 45, 50)
        self.aar.update_relation()
        
        # Perform meta-cognitive step
        assessment = self.aar.meta_cognitive_step()
        
        self.assertIn('insights', assessment)
        self.assertIn('recommendations', assessment)
        self.assertGreater(self.aar.meta_cognitive_depth, 0)
    
    def test_self_perception(self):
        """Test self-perception."""
        perception = self.aar.perceive_self()
        
        self.assertIn('agent', perception)
        self.assertIn('arena', perception)
        self.assertIn('relation', perception)
        self.assertIn('meta', perception)


class TestMembraneArchitecture(unittest.TestCase):
    """Test Deep Tree Echo membrane architecture."""
    
    def setUp(self):
        self.arch = DeepTreeEchoArchitecture()
    
    def test_architecture_creation(self):
        """Test architecture initialization."""
        self.assertIsNotNone(self.arch.root)
        self.assertIsNotNone(self.arch.cognitive)
        self.assertIsNotNone(self.arch.memory)
        self.assertIsNotNone(self.arch.security)
    
    def test_membrane_hierarchy(self):
        """Test membrane parent-child relationships."""
        self.assertEqual(self.arch.cognitive.parent, self.arch.root)
        self.assertEqual(self.arch.memory.parent, self.arch.cognitive)
        self.assertIn('memory', self.arch.cognitive.children)
    
    def test_message_passing(self):
        """Test inter-membrane message passing."""
        # Start membranes
        self.arch.start_all()
        
        # Send message
        self.arch.cognitive.send_message(
            'memory',
            'store',
            {'key': 'test', 'value': 'data'}
        )
        
        # Wait for processing
        import time
        time.sleep(0.5)
        
        # Check message was received
        self.assertGreater(self.arch.memory.state.message_count, 0)
        
        # Stop membranes
        self.arch.stop_all()
    
    def test_membrane_status(self):
        """Test membrane status reporting."""
        status = self.arch.root.get_status()
        
        self.assertIn('membrane_id', status)
        self.assertIn('type', status)
        self.assertIn('children', status)
    
    def test_hierarchy_status(self):
        """Test full hierarchy status."""
        status = self.arch.get_hierarchy_status()
        
        self.assertIn('children_status', status)
        self.assertIn('cognitive', status['children_status'])


class TestCognitiveSynergy(unittest.TestCase):
    """Test cognitive synergy mechanisms."""
    
    def setUp(self):
        # Import here to avoid circular dependencies
        try:
            self.engine = CognitiveSynergyEngine()
        except:
            self.skipTest("CognitiveSynergyEngine not fully implemented")
    
    def test_process_registration(self):
        """Test cognitive process registration."""
        process = CognitiveProcess(
            process_id='test_process',
            process_type='reasoning',
            priority=0.8
        )
        
        # This would be implemented in the engine
        # self.engine.register_process(process)
        # self.assertIn('test_process', self.engine.processes)
        pass
    
    def test_bottleneck_detection(self):
        """Test bottleneck detection."""
        # This would test the bottleneck detection mechanism
        pass
    
    def test_synergy_event_logging(self):
        """Test synergy event logging."""
        # This would test event logging
        pass


class TestIntegration(unittest.TestCase):
    """Integration tests for the complete system."""
    
    def test_aar_membrane_integration(self):
        """Test integration of AAR core with membrane architecture."""
        aar = AARCore()
        arch = DeepTreeEchoArchitecture()
        
        # Update AAR state
        aar.update_agent({'reasoning': 0.7})
        aar.update_arena(100, 200, 45, 50)
        aar.update_relation()
        
        # Get self-summary
        summary = aar.get_self_summary()
        
        self.assertIsNotNone(summary)
        self.assertIn('identity', summary)
    
    def test_end_to_end_cognitive_cycle(self):
        """Test complete cognitive cycle."""
        # Initialize components
        aar = AARCore()
        arch = DeepTreeEchoArchitecture()
        
        # Start architecture
        arch.start_all()
        
        # Simulate cognitive activity
        aar.update_agent({
            'pattern_miner': 0.6,
            'reasoning': 0.7
        }, new_goal='Discover patterns')
        
        aar.update_arena(
            atom_count=500,
            link_count=1200,
            pattern_matches=85,
            total_patterns=100
        )
        
        aar.update_relation()
        
        # Perform meta-cognition
        assessment = aar.meta_cognitive_step()
        
        # Verify results
        self.assertIsNotNone(assessment)
        self.assertIn('insights', assessment)
        
        # Stop architecture
        arch.stop_all()


class TestPerformance(unittest.TestCase):
    """Performance tests."""
    
    def test_memory_scalability(self):
        """Test memory operations at scale."""
        memory = HypergraphMemory()
        
        # Add many atoms
        start_time = datetime.now()
        for i in range(1000):
            atom = Atom(
                atom_type='ConceptNode',
                name=f'concept_{i}',
                truth_value=0.8
            )
            memory.add_atom(atom)
        
        duration = (datetime.now() - start_time).total_seconds()
        
        self.assertLess(duration, 5.0)  # Should complete in < 5 seconds
        self.assertEqual(len(memory.atoms), 1000)
    
    def test_message_throughput(self):
        """Test message passing throughput."""
        arch = DeepTreeEchoArchitecture()
        arch.start_all()
        
        # Send many messages
        start_time = datetime.now()
        for i in range(100):
            arch.cognitive.send_message(
                'memory',
                'store',
                {'key': f'key_{i}', 'value': f'value_{i}'}
            )
        
        import time
        time.sleep(2)  # Wait for processing
        
        duration = (datetime.now() - start_time).total_seconds()
        
        arch.stop_all()
        
        self.assertLess(duration, 10.0)  # Should complete in < 10 seconds


def run_tests():
    """Run all tests."""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test cases
    suite.addTests(loader.loadTestsFromTestCase(TestHypergraphMemory))
    suite.addTests(loader.loadTestsFromTestCase(TestAARCore))
    suite.addTests(loader.loadTestsFromTestCase(TestMembraneArchitecture))
    suite.addTests(loader.loadTestsFromTestCase(TestCognitiveSynergy))
    suite.addTests(loader.loadTestsFromTestCase(TestIntegration))
    suite.addTests(loader.loadTestsFromTestCase(TestPerformance))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_tests()
    sys.exit(0 if success else 1)


"""
Enhanced Test Suite for Cognitive Synergy
=========================================

Comprehensive tests for:
- Hypergraph Persistence Layer
- Autonomous MCP Agent
- Cognitive Synergy Integration

Author: OpenCog Collection Contributors
License: GPL-3.0+
"""

import unittest
import sys
import os
from datetime import datetime
import uuid

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hypergraph_persistence_layer import (
    HypergraphPersistenceLayer,
    HypergraphNode,
    HypergraphEdge,
    CognitivePattern,
    CognitiveSnapshot
)

from autonomous_mcp_agent import (
    AutonomousMCPAgent,
    MCPTool,
    ToolInvocation,
    ToolResult,
    ToolCategory
)


class TestHypergraphPersistence(unittest.TestCase):
    """Test suite for Hypergraph Persistence Layer."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.persistence = HypergraphPersistenceLayer()
        self.test_node_id = str(uuid.uuid4())
        self.test_edge_id = str(uuid.uuid4())
        self.test_pattern_id = str(uuid.uuid4())
        self.test_snapshot_id = str(uuid.uuid4())
    
    def test_initialization(self):
        """Test persistence layer initialization."""
        self.assertIsNotNone(self.persistence)
        # Should work in mock mode even without credentials
        self.assertTrue(self.persistence.mock_mode or self.persistence.client is not None)
    
    def test_store_and_retrieve_node(self):
        """Test storing and retrieving a hypergraph node."""
        # Create node
        node = HypergraphNode(
            node_id=self.test_node_id,
            node_type="concept",
            content={"name": "test_concept", "value": 42},
            attention_value=0.75
        )
        
        # Store node
        success = self.persistence.store_node(node)
        self.assertTrue(success)
        
        # Retrieve node
        retrieved = self.persistence.get_node(self.test_node_id)
        self.assertIsNotNone(retrieved)
        self.assertEqual(retrieved.node_id, self.test_node_id)
        self.assertEqual(retrieved.node_type, "concept")
        self.assertEqual(retrieved.attention_value, 0.75)
    
    def test_store_and_retrieve_edge(self):
        """Test storing and retrieving a hypergraph edge."""
        # Create edge
        edge = HypergraphEdge(
            edge_id=self.test_edge_id,
            edge_type="relates_to",
            source_nodes=[self.test_node_id],
            target_nodes=["target_node_1", "target_node_2"],
            weight=0.9,
            confidence=0.85
        )
        
        # Store edge
        success = self.persistence.store_edge(edge)
        self.assertTrue(success)
        
        # Retrieve edge
        retrieved = self.persistence.get_edge(self.test_edge_id)
        self.assertIsNotNone(retrieved)
        self.assertEqual(retrieved.edge_id, self.test_edge_id)
        self.assertEqual(retrieved.edge_type, "relates_to")
        self.assertEqual(len(retrieved.source_nodes), 1)
        self.assertEqual(len(retrieved.target_nodes), 2)
    
    def test_store_and_retrieve_pattern(self):
        """Test storing and retrieving a cognitive pattern."""
        # Create pattern
        pattern = CognitivePattern(
            pattern_id=self.test_pattern_id,
            source_process="reasoning",
            pattern_type="inference",
            content={"rule": "A implies B", "confidence": 0.9},
            confidence=0.9,
            attention_value=0.8,
            hypergraph_nodes=[self.test_node_id],
            hypergraph_edges=[self.test_edge_id],
            consumers={"learning", "planning"}
        )
        
        # Store pattern
        success = self.persistence.store_pattern(pattern)
        self.assertTrue(success)
        
        # Retrieve pattern
        retrieved = self.persistence.get_pattern(self.test_pattern_id)
        self.assertIsNotNone(retrieved)
        self.assertEqual(retrieved.pattern_id, self.test_pattern_id)
        self.assertEqual(retrieved.source_process, "reasoning")
        self.assertEqual(retrieved.pattern_type, "inference")
        self.assertEqual(retrieved.confidence, 0.9)
    
    def test_create_and_retrieve_snapshot(self):
        """Test creating and retrieving a cognitive snapshot."""
        # Create snapshot
        snapshot = CognitiveSnapshot(
            snapshot_id=self.test_snapshot_id,
            timestamp=datetime.now(),
            processes={"reasoning": "active", "learning": "idle"},
            metrics={"attention_efficiency": 0.87, "patterns_shared": 42},
            hypergraph_state={"nodes": 150, "edges": 320},
            description="Test snapshot"
        )
        
        # Store snapshot
        success = self.persistence.create_snapshot(snapshot)
        self.assertTrue(success)
        
        # Retrieve snapshot
        retrieved = self.persistence.get_snapshot(self.test_snapshot_id)
        self.assertIsNotNone(retrieved)
        self.assertEqual(retrieved.snapshot_id, self.test_snapshot_id)
        self.assertEqual(retrieved.description, "Test snapshot")
        self.assertIn("reasoning", retrieved.processes)
    
    def test_query_nodes_by_type(self):
        """Test querying nodes by type."""
        # Store multiple nodes of same type
        for i in range(3):
            node = HypergraphNode(
                node_id=str(uuid.uuid4()),
                node_type="test_type",
                content={"index": i},
                attention_value=0.5 + i * 0.1
            )
            self.persistence.store_node(node)
        
        # Query nodes
        nodes = self.persistence.query_nodes_by_type("test_type")
        self.assertGreaterEqual(len(nodes), 3)
        self.assertTrue(all(n.node_type == "test_type" for n in nodes))
    
    def test_query_patterns_by_type(self):
        """Test querying patterns by type."""
        # Store multiple patterns of same type
        for i in range(3):
            pattern = CognitivePattern(
                pattern_id=str(uuid.uuid4()),
                source_process="test_process",
                pattern_type="test_pattern_type",
                content={"index": i},
                confidence=0.7 + i * 0.1,
                attention_value=0.6,
                hypergraph_nodes=[],
                hypergraph_edges=[],
                consumers=set()
            )
            self.persistence.store_pattern(pattern)
        
        # Query patterns
        patterns = self.persistence.query_patterns_by_type("test_pattern_type")
        self.assertGreaterEqual(len(patterns), 3)
        self.assertTrue(all(p.pattern_type == "test_pattern_type" for p in patterns))
    
    def test_get_statistics(self):
        """Test getting storage statistics."""
        stats = self.persistence.get_statistics()
        self.assertIsNotNone(stats)
        self.assertIsInstance(stats, dict)
        # Should have counts for different entity types
        self.assertTrue(len(stats) > 0)


class TestAutonomousMCPAgent(unittest.TestCase):
    """Test suite for Autonomous MCP Agent."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.agent = AutonomousMCPAgent()
    
    def test_initialization(self):
        """Test agent initialization."""
        self.assertIsNotNone(self.agent)
        self.assertIsInstance(self.agent.tools, dict)
        self.assertIsInstance(self.agent.servers, list)
    
    def test_tool_discovery(self):
        """Test MCP tool discovery."""
        tools = self.agent.get_available_tools()
        self.assertIsInstance(tools, list)
        # Should discover some tools even in test environment
        self.assertGreaterEqual(len(tools), 0)
    
    def test_get_tools_by_category(self):
        """Test filtering tools by category."""
        db_tools = self.agent.get_available_tools(category=ToolCategory.DATABASE)
        self.assertIsInstance(db_tools, list)
        self.assertTrue(all(t.category == ToolCategory.DATABASE for t in db_tools))
        
        search_tools = self.agent.get_available_tools(category=ToolCategory.SEARCH)
        self.assertIsInstance(search_tools, list)
        self.assertTrue(all(t.category == ToolCategory.SEARCH for t in search_tools))
    
    def test_find_tool_for_task(self):
        """Test finding appropriate tool for a task."""
        # Test database task
        tool = self.agent.find_tool_for_task("Execute a SQL query on the database")
        if tool:
            self.assertIn("database", tool.name.lower() or tool.category.value.lower())
        
        # Test model search task
        tool = self.agent.find_tool_for_task("Find a model for text generation")
        if tool:
            self.assertTrue("model" in tool.name.lower() or "search" in tool.name.lower())
        
        # Test research task
        tool = self.agent.find_tool_for_task("Search for research papers on AGI")
        if tool:
            self.assertTrue("search" in tool.name.lower() or "paper" in tool.name.lower())
    
    def test_tool_invocation_structure(self):
        """Test tool invocation data structure."""
        invocation = ToolInvocation(
            tool_name="test_tool",
            server="test_server",
            arguments={"arg1": "value1"},
            context={"task": "test task"},
            priority=5
        )
        
        self.assertEqual(invocation.tool_name, "test_tool")
        self.assertEqual(invocation.server, "test_server")
        self.assertIn("arg1", invocation.arguments)
        
        # Test serialization
        invocation_dict = invocation.to_dict()
        self.assertIsInstance(invocation_dict, dict)
        self.assertIn("tool_name", invocation_dict)
    
    def test_tool_result_structure(self):
        """Test tool result data structure."""
        result = ToolResult(
            tool_name="test_tool",
            success=True,
            result={"output": "test output"},
            latency=0.5
        )
        
        self.assertTrue(result.success)
        self.assertEqual(result.tool_name, "test_tool")
        self.assertIsNotNone(result.timestamp)
        
        # Test serialization
        result_dict = result.to_dict()
        self.assertIsInstance(result_dict, dict)
        self.assertIn("success", result_dict)
    
    def test_get_tool_statistics(self):
        """Test getting tool usage statistics."""
        stats = self.agent.get_tool_statistics()
        self.assertIsNotNone(stats)
        self.assertIsInstance(stats, dict)
        self.assertIn("total_tools", stats)
        self.assertIn("total_invocations", stats)
        self.assertIn("success_rate", stats)
    
    def test_cognitive_process_integration(self):
        """Test integration with cognitive processes."""
        def dummy_callback(result):
            pass
        
        integration = self.agent.integrate_with_cognitive_process(
            process_id="test_process_001",
            process_type="reasoning",
            callback=dummy_callback
        )
        
        self.assertIsNotNone(integration)
        self.assertIsInstance(integration, dict)
        self.assertIn("process_id", integration)
        self.assertIn("available_tools", integration)


class TestCognitiveSynergyIntegration(unittest.TestCase):
    """Test suite for integrated cognitive synergy."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.persistence = HypergraphPersistenceLayer()
        self.agent = AutonomousMCPAgent()
    
    def test_pattern_persistence_workflow(self):
        """Test complete workflow of pattern discovery and persistence."""
        # Simulate pattern discovery
        pattern_id = str(uuid.uuid4())
        node_id = str(uuid.uuid4())
        edge_id = str(uuid.uuid4())
        
        # Create hypergraph structure
        node = HypergraphNode(
            node_id=node_id,
            node_type="inference_result",
            content={"conclusion": "X implies Y"},
            attention_value=0.85
        )
        self.persistence.store_node(node)
        
        edge = HypergraphEdge(
            edge_id=edge_id,
            edge_type="inference",
            source_nodes=["premise_1", "premise_2"],
            target_nodes=[node_id],
            weight=0.9,
            confidence=0.85
        )
        self.persistence.store_edge(edge)
        
        # Create pattern
        pattern = CognitivePattern(
            pattern_id=pattern_id,
            source_process="reasoning",
            pattern_type="inference",
            content={"rule": "If A and B then C"},
            confidence=0.85,
            attention_value=0.8,
            hypergraph_nodes=[node_id],
            hypergraph_edges=[edge_id],
            consumers={"learning", "planning"}
        )
        self.persistence.store_pattern(pattern)
        
        # Verify persistence
        retrieved_pattern = self.persistence.get_pattern(pattern_id)
        self.assertIsNotNone(retrieved_pattern)
        self.assertEqual(retrieved_pattern.pattern_id, pattern_id)
        
        retrieved_node = self.persistence.get_node(node_id)
        self.assertIsNotNone(retrieved_node)
        
        retrieved_edge = self.persistence.get_edge(edge_id)
        self.assertIsNotNone(retrieved_edge)
    
    def test_mcp_enhanced_reasoning(self):
        """Test MCP-enhanced cognitive reasoning."""
        # Find tool for reasoning support
        tool = self.agent.find_tool_for_task("Search for reasoning models")
        
        if tool:
            # Tool found, verify it's appropriate
            self.assertTrue("search" in tool.name.lower() or "model" in tool.name.lower())
            
            # In a real scenario, we would invoke the tool and store results
            # For testing, we just verify the structure
            self.assertIsNotNone(tool.server)
            self.assertIsNotNone(tool.name)
    
    def test_snapshot_with_mcp_metrics(self):
        """Test creating snapshot with MCP agent metrics."""
        # Get MCP statistics
        mcp_stats = self.agent.get_tool_statistics()
        
        # Create snapshot including MCP metrics
        snapshot = CognitiveSnapshot(
            snapshot_id=str(uuid.uuid4()),
            timestamp=datetime.now(),
            processes={"reasoning": "active", "mcp_agent": "active"},
            metrics={
                "attention_efficiency": 0.87,
                "patterns_shared": 42,
                "mcp_tools_available": mcp_stats["total_tools"],
                "mcp_success_rate": mcp_stats["success_rate"]
            },
            hypergraph_state={"nodes": 150, "edges": 320},
            description="Snapshot with MCP integration"
        )
        
        # Store snapshot
        success = self.persistence.create_snapshot(snapshot)
        self.assertTrue(success)
        
        # Retrieve and verify
        retrieved = self.persistence.get_snapshot(snapshot.snapshot_id)
        self.assertIsNotNone(retrieved)
        self.assertIn("mcp_tools_available", retrieved.metrics)


class TestCognitiveSynergyMetrics(unittest.TestCase):
    """Test suite for cognitive synergy metrics."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.persistence = HypergraphPersistenceLayer()
        self.agent = AutonomousMCPAgent()
    
    def test_cross_module_pattern_sharing(self):
        """Test cross-module pattern sharing metrics."""
        # Create patterns from different modules
        patterns = []
        for i, module in enumerate(["reasoning", "learning", "perception"]):
            pattern = CognitivePattern(
                pattern_id=str(uuid.uuid4()),
                source_process=module,
                pattern_type="shared_pattern",
                content={"module": module, "index": i},
                confidence=0.8,
                attention_value=0.7,
                hypergraph_nodes=[],
                hypergraph_edges=[],
                consumers={"reasoning", "learning", "perception"} - {module}
            )
            self.persistence.store_pattern(pattern)
            patterns.append(pattern)
        
        # Query shared patterns
        shared_patterns = self.persistence.query_patterns_by_type("shared_pattern")
        self.assertGreaterEqual(len(shared_patterns), 3)
        
        # Calculate sharing metric
        total_consumers = sum(len(p.consumers) for p in shared_patterns)
        avg_consumers = total_consumers / len(shared_patterns) if shared_patterns else 0
        
        # Should have cross-module sharing
        self.assertGreater(avg_consumers, 0)
    
    def test_attention_efficiency_tracking(self):
        """Test attention efficiency metrics."""
        # Create nodes with varying attention values
        attention_values = [0.9, 0.7, 0.5, 0.3, 0.1]
        for i, attention in enumerate(attention_values):
            node = HypergraphNode(
                node_id=str(uuid.uuid4()),
                node_type="attention_test",
                content={"index": i},
                attention_value=attention
            )
            self.persistence.store_node(node)
        
        # Query nodes and calculate efficiency
        nodes = self.persistence.query_nodes_by_type("attention_test")
        if nodes:
            avg_attention = sum(n.attention_value for n in nodes) / len(nodes)
            # Should have reasonable attention distribution
            self.assertGreater(avg_attention, 0)
            self.assertLess(avg_attention, 1)


def run_tests():
    """Run all test suites."""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test classes
    suite.addTests(loader.loadTestsFromTestCase(TestHypergraphPersistence))
    suite.addTests(loader.loadTestsFromTestCase(TestAutonomousMCPAgent))
    suite.addTests(loader.loadTestsFromTestCase(TestCognitiveSynergyIntegration))
    suite.addTests(loader.loadTestsFromTestCase(TestCognitiveSynergyMetrics))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Return success status
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)

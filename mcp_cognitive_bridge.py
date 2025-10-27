"""
MCP Cognitive Bridge
===================

This module bridges Model Context Protocol (MCP) servers with the cognitive
architecture, enabling external tool integration for enhanced cognitive capabilities.

Features:
- MCP server discovery and capability mapping
- Cognitive task to MCP tool routing
- Result integration into hypergraph
- Neon database operations for knowledge storage
- Hugging Face model discovery and evaluation

Author: OpenCog Collection Contributors
License: GPL-3.0+
"""

import os
import logging
import subprocess
import json
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MCPServerType(Enum):
    """Types of MCP servers available."""
    NEON = "neon"
    HUGGING_FACE = "hugging-face"


@dataclass
class MCPTool:
    """Representation of an MCP tool."""
    name: str
    server: str
    description: str
    input_schema: Dict[str, Any]
    capabilities: List[str] = field(default_factory=list)


@dataclass
class MCPToolResult:
    """Result from MCP tool execution."""
    tool_name: str
    success: bool
    result: Any
    error: Optional[str] = None
    execution_time: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)


class MCPClient:
    """
    Client for interacting with MCP servers via manus-mcp-cli.
    """
    
    def __init__(self):
        """Initialize MCP client."""
        self.available_servers = ["neon", "hugging-face"]
        self.tool_cache: Dict[str, List[MCPTool]] = {}
        logger.info("MCPClient initialized")
    
    def _run_mcp_command(self, args: List[str]) -> Tuple[bool, str]:
        """Run manus-mcp-cli command."""
        try:
            cmd = ["manus-mcp-cli"] + args
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.returncode == 0:
                return True, result.stdout
            else:
                logger.error(f"MCP command failed: {result.stderr}")
                return False, result.stderr
        except subprocess.TimeoutExpired:
            logger.error("MCP command timed out")
            return False, "Command timed out"
        except Exception as e:
            logger.error(f"MCP command error: {e}")
            return False, str(e)
    
    def discover_tools(self, server: str) -> List[MCPTool]:
        """Discover available tools from an MCP server."""
        if server in self.tool_cache:
            return self.tool_cache[server]
        
        logger.info(f"Discovering tools from {server} server...")
        success, output = self._run_mcp_command(["tool", "list", "--server", server])
        
        if not success:
            logger.error(f"Failed to discover tools from {server}")
            return []
        
        # Parse tool list (assuming JSON output)
        try:
            tools_data = json.loads(output)
            tools = []
            
            for tool_data in tools_data.get("tools", []):
                tool = MCPTool(
                    name=tool_data.get("name", ""),
                    server=server,
                    description=tool_data.get("description", ""),
                    input_schema=tool_data.get("inputSchema", {}),
                    capabilities=self._infer_capabilities(tool_data)
                )
                tools.append(tool)
            
            self.tool_cache[server] = tools
            logger.info(f"Discovered {len(tools)} tools from {server}")
            return tools
        except json.JSONDecodeError:
            logger.error(f"Failed to parse tool list from {server}")
            return []
    
    def _infer_capabilities(self, tool_data: Dict[str, Any]) -> List[str]:
        """Infer cognitive capabilities from tool metadata."""
        capabilities = []
        name = tool_data.get("name", "").lower()
        description = tool_data.get("description", "").lower()
        
        # Database capabilities
        if any(kw in name or kw in description for kw in ["database", "query", "sql", "table"]):
            capabilities.append("database_access")
        
        # Model capabilities
        if any(kw in name or kw in description for kw in ["model", "inference", "predict"]):
            capabilities.append("model_inference")
        
        # Search capabilities
        if any(kw in name or kw in description for kw in ["search", "find", "discover"]):
            capabilities.append("knowledge_search")
        
        # Data capabilities
        if any(kw in name or kw in description for kw in ["dataset", "data", "download"]):
            capabilities.append("data_access")
        
        return capabilities
    
    def call_tool(self, tool_name: str, server: str, inputs: Dict[str, Any]) -> MCPToolResult:
        """Call an MCP tool with given inputs."""
        start_time = datetime.now()
        
        logger.info(f"Calling {tool_name} on {server} server...")
        
        # Convert inputs to JSON string
        input_json = json.dumps(inputs)
        
        success, output = self._run_mcp_command([
            "tool", "call", tool_name,
            "--server", server,
            "--input", input_json
        ])
        
        execution_time = (datetime.now() - start_time).total_seconds()
        
        if success:
            try:
                result_data = json.loads(output)
                return MCPToolResult(
                    tool_name=tool_name,
                    success=True,
                    result=result_data,
                    execution_time=execution_time
                )
            except json.JSONDecodeError:
                return MCPToolResult(
                    tool_name=tool_name,
                    success=True,
                    result=output,
                    execution_time=execution_time
                )
        else:
            return MCPToolResult(
                tool_name=tool_name,
                success=False,
                result=None,
                error=output,
                execution_time=execution_time
            )


class NeonCognitiveInterface:
    """
    Interface for Neon database operations in cognitive context.
    """
    
    def __init__(self, mcp_client: MCPClient):
        """Initialize Neon interface."""
        self.mcp_client = mcp_client
        self.server = "neon"
        logger.info("NeonCognitiveInterface initialized")
    
    def execute_query(self, query: str) -> MCPToolResult:
        """Execute SQL query on Neon database."""
        # Discover query tool
        tools = self.mcp_client.discover_tools(self.server)
        query_tool = next((t for t in tools if "query" in t.name.lower()), None)
        
        if not query_tool:
            return MCPToolResult(
                tool_name="neon_query",
                success=False,
                result=None,
                error="Query tool not found"
            )
        
        return self.mcp_client.call_tool(
            query_tool.name,
            self.server,
            {"query": query}
        )
    
    def store_cognitive_pattern(self, pattern_data: Dict[str, Any]) -> MCPToolResult:
        """Store a discovered cognitive pattern in Neon."""
        # Create table if not exists
        create_table_query = """
        CREATE TABLE IF NOT EXISTS cognitive_patterns (
            pattern_id VARCHAR(255) PRIMARY KEY,
            pattern_type VARCHAR(100),
            structure JSONB,
            frequency INTEGER,
            confidence FLOAT,
            source_paradigm VARCHAR(100),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """
        
        self.execute_query(create_table_query)
        
        # Insert pattern
        insert_query = f"""
        INSERT INTO cognitive_patterns (pattern_id, pattern_type, structure, frequency, confidence, source_paradigm)
        VALUES (
            '{pattern_data.get("pattern_id")}',
            '{pattern_data.get("pattern_type")}',
            '{json.dumps(pattern_data.get("structure"))}'::jsonb,
            {pattern_data.get("frequency", 1)},
            {pattern_data.get("confidence", 1.0)},
            '{pattern_data.get("source_paradigm", "unknown")}'
        )
        ON CONFLICT (pattern_id) DO UPDATE SET
            frequency = cognitive_patterns.frequency + 1,
            confidence = EXCLUDED.confidence;
        """
        
        return self.execute_query(insert_query)
    
    def retrieve_patterns(self, pattern_type: Optional[str] = None, min_confidence: float = 0.5) -> MCPToolResult:
        """Retrieve cognitive patterns from Neon."""
        query = f"""
        SELECT * FROM cognitive_patterns
        WHERE confidence >= {min_confidence}
        """
        
        if pattern_type:
            query += f" AND pattern_type = '{pattern_type}'"
        
        query += " ORDER BY confidence DESC, frequency DESC LIMIT 100;"
        
        return self.execute_query(query)


class HuggingFaceCognitiveInterface:
    """
    Interface for Hugging Face model discovery and evaluation.
    """
    
    def __init__(self, mcp_client: MCPClient):
        """Initialize Hugging Face interface."""
        self.mcp_client = mcp_client
        self.server = "hugging-face"
        logger.info("HuggingFaceCognitiveInterface initialized")
    
    def search_models(self, query: str, task: Optional[str] = None) -> MCPToolResult:
        """Search for models on Hugging Face."""
        tools = self.mcp_client.discover_tools(self.server)
        search_tool = next((t for t in tools if "search" in t.name.lower() and "model" in t.name.lower()), None)
        
        if not search_tool:
            return MCPToolResult(
                tool_name="hf_search_models",
                success=False,
                result=None,
                error="Model search tool not found"
            )
        
        inputs = {"query": query}
        if task:
            inputs["task"] = task
        
        return self.mcp_client.call_tool(search_tool.name, self.server, inputs)
    
    def get_model_info(self, model_id: str) -> MCPToolResult:
        """Get detailed information about a model."""
        tools = self.mcp_client.discover_tools(self.server)
        info_tool = next((t for t in tools if "info" in t.name.lower() or "get" in t.name.lower()), None)
        
        if not info_tool:
            return MCPToolResult(
                tool_name="hf_model_info",
                success=False,
                result=None,
                error="Model info tool not found"
            )
        
        return self.mcp_client.call_tool(
            info_tool.name,
            self.server,
            {"model_id": model_id}
        )
    
    def search_datasets(self, query: str) -> MCPToolResult:
        """Search for datasets on Hugging Face."""
        tools = self.mcp_client.discover_tools(self.server)
        search_tool = next((t for t in tools if "search" in t.name.lower() and "dataset" in t.name.lower()), None)
        
        if not search_tool:
            return MCPToolResult(
                tool_name="hf_search_datasets",
                success=False,
                result=None,
                error="Dataset search tool not found"
            )
        
        return self.mcp_client.call_tool(
            search_tool.name,
            self.server,
            {"query": query}
        )


class MCPCognitiveBridge:
    """
    Main bridge between MCP servers and cognitive architecture.
    Routes cognitive tasks to appropriate MCP tools.
    """
    
    def __init__(self):
        """Initialize MCP cognitive bridge."""
        self.mcp_client = MCPClient()
        self.neon_interface = NeonCognitiveInterface(self.mcp_client)
        self.hf_interface = HuggingFaceCognitiveInterface(self.mcp_client)
        
        # Discover all available tools
        self.all_tools: Dict[str, List[MCPTool]] = {}
        for server in self.mcp_client.available_servers:
            self.all_tools[server] = self.mcp_client.discover_tools(server)
        
        logger.info("MCPCognitiveBridge initialized")
    
    def route_cognitive_task(self, task_type: str, task_data: Dict[str, Any]) -> MCPToolResult:
        """Route a cognitive task to the appropriate MCP tool."""
        logger.info(f"Routing cognitive task: {task_type}")
        
        if task_type == "store_pattern":
            return self.neon_interface.store_cognitive_pattern(task_data)
        
        elif task_type == "retrieve_patterns":
            return self.neon_interface.retrieve_patterns(
                pattern_type=task_data.get("pattern_type"),
                min_confidence=task_data.get("min_confidence", 0.5)
            )
        
        elif task_type == "search_models":
            return self.hf_interface.search_models(
                query=task_data.get("query", ""),
                task=task_data.get("task")
            )
        
        elif task_type == "search_datasets":
            return self.hf_interface.search_datasets(
                query=task_data.get("query", "")
            )
        
        elif task_type == "execute_query":
            return self.neon_interface.execute_query(task_data.get("query", ""))
        
        else:
            return MCPToolResult(
                tool_name="unknown",
                success=False,
                result=None,
                error=f"Unknown task type: {task_type}"
            )
    
    def get_capabilities(self) -> Dict[str, List[str]]:
        """Get all available cognitive capabilities from MCP tools."""
        capabilities = {}
        
        for server, tools in self.all_tools.items():
            server_caps = []
            for tool in tools:
                server_caps.extend(tool.capabilities)
            capabilities[server] = list(set(server_caps))
        
        return capabilities
    
    def enhance_cognitive_module(self, module_name: str, required_capabilities: List[str]) -> List[MCPTool]:
        """Find MCP tools that can enhance a cognitive module."""
        matching_tools = []
        
        for server, tools in self.all_tools.items():
            for tool in tools:
                if any(cap in tool.capabilities for cap in required_capabilities):
                    matching_tools.append(tool)
        
        logger.info(f"Found {len(matching_tools)} tools to enhance {module_name}")
        return matching_tools


# Example usage
if __name__ == "__main__":
    # Initialize bridge
    bridge = MCPCognitiveBridge()
    
    # Get available capabilities
    capabilities = bridge.get_capabilities()
    print("Available capabilities:", json.dumps(capabilities, indent=2))
    
    # Store a cognitive pattern
    pattern_data = {
        "pattern_id": "pattern_001",
        "pattern_type": "structural",
        "structure": {"nodes": 5, "edges": 8},
        "frequency": 10,
        "confidence": 0.85,
        "source_paradigm": "neural_symbolic"
    }
    
    result = bridge.route_cognitive_task("store_pattern", pattern_data)
    print(f"Store pattern result: {result.success}")
    
    # Search for models
    search_result = bridge.route_cognitive_task(
        "search_models",
        {"query": "cognitive reasoning", "task": "text-generation"}
    )
    print(f"Model search result: {search_result.success}")


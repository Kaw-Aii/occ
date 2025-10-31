"""
Autonomous MCP Agent for OpenCog Collection
===========================================

This module provides an autonomous agent that integrates Model Context Protocol (MCP)
tools with cognitive processes. It enables:

1. Automatic discovery of available MCP tools
2. Intelligent tool selection and invocation
3. Integration with cognitive orchestrator
4. Query optimization using Neon database
5. Model discovery via Hugging Face

Author: OpenCog Collection Contributors
License: GPL-3.0+
"""

import logging
import subprocess
import json
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ToolCategory(Enum):
    """Categories of MCP tools."""
    DATABASE = "database"
    MODEL = "model"
    SEARCH = "search"
    COMPUTATION = "computation"
    STORAGE = "storage"
    COMMUNICATION = "communication"


@dataclass
class MCPTool:
    """Represents an MCP tool."""
    name: str
    server: str
    category: ToolCategory
    description: str
    parameters: Dict[str, Any]
    capabilities: List[str]
    usage_count: int = 0
    success_rate: float = 1.0
    average_latency: float = 0.0
    last_used: Optional[datetime] = None
    
    def update_metrics(self, success: bool, latency: float):
        """Update tool usage metrics."""
        self.usage_count += 1
        self.success_rate = ((self.success_rate * (self.usage_count - 1)) + (1.0 if success else 0.0)) / self.usage_count
        self.average_latency = ((self.average_latency * (self.usage_count - 1)) + latency) / self.usage_count
        self.last_used = datetime.now()


@dataclass
class ToolInvocation:
    """Represents a tool invocation request."""
    tool_name: str
    server: str
    arguments: Dict[str, Any]
    context: Dict[str, Any] = field(default_factory=dict)
    priority: int = 5
    timeout: int = 30
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "tool_name": self.tool_name,
            "server": self.server,
            "arguments": self.arguments,
            "context": self.context,
            "priority": self.priority,
            "timeout": self.timeout
        }


@dataclass
class ToolResult:
    """Represents the result of a tool invocation."""
    tool_name: str
    success: bool
    result: Any
    error: Optional[str] = None
    latency: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "tool_name": self.tool_name,
            "success": self.success,
            "result": self.result,
            "error": self.error,
            "latency": self.latency,
            "timestamp": self.timestamp.isoformat()
        }


class AutonomousMCPAgent:
    """
    Autonomous agent for MCP tool integration with cognitive processes.
    
    This agent:
    - Discovers available MCP tools
    - Intelligently selects appropriate tools for tasks
    - Integrates tools with cognitive processes
    - Optimizes tool usage based on performance metrics
    """
    
    def __init__(self):
        """Initialize the autonomous MCP agent."""
        self.tools: Dict[str, MCPTool] = {}
        self.servers: List[str] = ["neon", "hugging-face"]
        self.tool_cache: Dict[str, Any] = {}
        self.invocation_history: List[ToolInvocation] = []
        self.result_history: List[ToolResult] = []
        
        logger.info("Autonomous MCP Agent initialized")
        self._discover_all_tools()
    
    def _discover_all_tools(self):
        """Discover all available MCP tools from configured servers."""
        logger.info("Discovering MCP tools...")
        
        for server in self.servers:
            try:
                tools = self._discover_server_tools(server)
                logger.info(f"Discovered {len(tools)} tools from {server} server")
                
                for tool in tools:
                    self.tools[f"{server}:{tool.name}"] = tool
            except Exception as e:
                logger.error(f"Failed to discover tools from {server}: {e}")
        
        logger.info(f"Total tools discovered: {len(self.tools)}")
    
    def _discover_server_tools(self, server: str) -> List[MCPTool]:
        """Discover tools from a specific MCP server."""
        try:
            # Execute manus-mcp-cli to list tools
            cmd = ["manus-mcp-cli", "tool", "list", "--server", server]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            
            if result.returncode != 0:
                logger.warning(f"Failed to list tools from {server}: {result.stderr}")
                return []
            
            # Parse the output to extract tool information
            tools = self._parse_tool_list(server, result.stdout)
            return tools
        except subprocess.TimeoutExpired:
            logger.error(f"Timeout while discovering tools from {server}")
            return []
        except Exception as e:
            logger.error(f"Error discovering tools from {server}: {e}")
            return []
    
    def _parse_tool_list(self, server: str, output: str) -> List[MCPTool]:
        """Parse tool list output from manus-mcp-cli."""
        tools = []
        
        # Define known tools for each server
        if server == "neon":
            tools.extend([
                MCPTool(
                    name="list_projects",
                    server=server,
                    category=ToolCategory.DATABASE,
                    description="List all Neon database projects",
                    parameters={},
                    capabilities=["database", "query", "management"]
                ),
                MCPTool(
                    name="create_project",
                    server=server,
                    category=ToolCategory.DATABASE,
                    description="Create a new Neon database project",
                    parameters={"project_name": "string"},
                    capabilities=["database", "management"]
                ),
                MCPTool(
                    name="execute_query",
                    server=server,
                    category=ToolCategory.DATABASE,
                    description="Execute SQL query on Neon database",
                    parameters={"project_id": "string", "sql": "string"},
                    capabilities=["database", "query", "data_access"]
                ),
                MCPTool(
                    name="create_branch",
                    server=server,
                    category=ToolCategory.DATABASE,
                    description="Create a database branch for experimentation",
                    parameters={"project_id": "string", "branch_name": "string"},
                    capabilities=["database", "versioning"]
                )
            ])
        elif server == "hugging-face":
            tools.extend([
                MCPTool(
                    name="search_models",
                    server=server,
                    category=ToolCategory.SEARCH,
                    description="Search for models on Hugging Face Hub",
                    parameters={"query": "string", "filter": "string"},
                    capabilities=["model", "search", "discovery"]
                ),
                MCPTool(
                    name="get_model_info",
                    server=server,
                    category=ToolCategory.MODEL,
                    description="Get detailed information about a model",
                    parameters={"model_id": "string"},
                    capabilities=["model", "metadata"]
                ),
                MCPTool(
                    name="search_datasets",
                    server=server,
                    category=ToolCategory.SEARCH,
                    description="Search for datasets on Hugging Face Hub",
                    parameters={"query": "string"},
                    capabilities=["dataset", "search", "discovery"]
                ),
                MCPTool(
                    name="search_papers",
                    server=server,
                    category=ToolCategory.SEARCH,
                    description="Search for research papers",
                    parameters={"query": "string"},
                    capabilities=["research", "search", "discovery"]
                )
            ])
        
        return tools
    
    def get_available_tools(self, category: Optional[ToolCategory] = None) -> List[MCPTool]:
        """Get list of available tools, optionally filtered by category."""
        if category:
            return [tool for tool in self.tools.values() if tool.category == category]
        return list(self.tools.values())
    
    def find_tool_for_task(self, task_description: str, context: Dict[str, Any] = None) -> Optional[MCPTool]:
        """
        Find the most appropriate tool for a given task.
        
        Uses simple keyword matching and context analysis.
        """
        task_lower = task_description.lower()
        context = context or {}
        
        # Keyword-based tool selection
        if "database" in task_lower or "query" in task_lower or "sql" in task_lower:
            # Prefer Neon database tools
            db_tools = [t for t in self.tools.values() if t.category == ToolCategory.DATABASE]
            if "execute" in task_lower or "query" in task_lower:
                return next((t for t in db_tools if "execute" in t.name), None)
            return db_tools[0] if db_tools else None
        
        if "model" in task_lower or "hugging face" in task_lower or "ml" in task_lower:
            # Prefer Hugging Face model tools
            model_tools = [t for t in self.tools.values() if t.category == ToolCategory.MODEL or "model" in t.name]
            return model_tools[0] if model_tools else None
        
        if "search" in task_lower or "find" in task_lower or "discover" in task_lower:
            # Prefer search tools
            search_tools = [t for t in self.tools.values() if t.category == ToolCategory.SEARCH]
            if "paper" in task_lower or "research" in task_lower:
                return next((t for t in search_tools if "paper" in t.name), None)
            if "dataset" in task_lower:
                return next((t for t in search_tools if "dataset" in t.name), None)
            return search_tools[0] if search_tools else None
        
        # Default: return most reliable tool
        if self.tools:
            sorted_tools = sorted(self.tools.values(), key=lambda t: t.success_rate, reverse=True)
            return sorted_tools[0]
        
        return None
    
    def invoke_tool(self, invocation: ToolInvocation) -> ToolResult:
        """
        Invoke an MCP tool with the given parameters.
        
        Args:
            invocation: Tool invocation request
            
        Returns:
            Tool result with success status and output
        """
        start_time = datetime.now()
        
        try:
            # Prepare arguments as JSON
            args_json = json.dumps(invocation.arguments)
            
            # Execute manus-mcp-cli
            cmd = [
                "manus-mcp-cli",
                "tool",
                "call",
                invocation.tool_name,
                "--server",
                invocation.server,
                "--input",
                args_json
            ]
            
            logger.info(f"Invoking tool: {invocation.server}:{invocation.tool_name}")
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=invocation.timeout
            )
            
            latency = (datetime.now() - start_time).total_seconds()
            
            if result.returncode == 0:
                # Parse result
                try:
                    output = json.loads(result.stdout) if result.stdout else {"raw": result.stdout}
                except json.JSONDecodeError:
                    output = {"raw": result.stdout}
                
                tool_result = ToolResult(
                    tool_name=invocation.tool_name,
                    success=True,
                    result=output,
                    latency=latency
                )
                
                # Update tool metrics
                tool_key = f"{invocation.server}:{invocation.tool_name}"
                if tool_key in self.tools:
                    self.tools[tool_key].update_metrics(True, latency)
                
                logger.info(f"Tool invocation successful: {invocation.tool_name} ({latency:.2f}s)")
                return tool_result
            else:
                error_msg = result.stderr or "Unknown error"
                tool_result = ToolResult(
                    tool_name=invocation.tool_name,
                    success=False,
                    result=None,
                    error=error_msg,
                    latency=latency
                )
                
                # Update tool metrics
                tool_key = f"{invocation.server}:{invocation.tool_name}"
                if tool_key in self.tools:
                    self.tools[tool_key].update_metrics(False, latency)
                
                logger.error(f"Tool invocation failed: {invocation.tool_name} - {error_msg}")
                return tool_result
        
        except subprocess.TimeoutExpired:
            latency = invocation.timeout
            tool_result = ToolResult(
                tool_name=invocation.tool_name,
                success=False,
                result=None,
                error="Timeout expired",
                latency=latency
            )
            
            # Update tool metrics
            tool_key = f"{invocation.server}:{invocation.tool_name}"
            if tool_key in self.tools:
                self.tools[tool_key].update_metrics(False, latency)
            
            logger.error(f"Tool invocation timeout: {invocation.tool_name}")
            return tool_result
        
        except Exception as e:
            latency = (datetime.now() - start_time).total_seconds()
            tool_result = ToolResult(
                tool_name=invocation.tool_name,
                success=False,
                result=None,
                error=str(e),
                latency=latency
            )
            
            logger.error(f"Tool invocation error: {invocation.tool_name} - {e}")
            return tool_result
        
        finally:
            # Record invocation and result
            self.invocation_history.append(invocation)
            if 'tool_result' in locals():
                self.result_history.append(tool_result)
    
    def autonomous_task_execution(self, task_description: str, context: Dict[str, Any] = None) -> ToolResult:
        """
        Autonomously execute a task by finding and invoking the appropriate tool.
        
        Args:
            task_description: Natural language description of the task
            context: Additional context for task execution
            
        Returns:
            Tool result
        """
        logger.info(f"Autonomous task execution: {task_description}")
        
        # Find appropriate tool
        tool = self.find_tool_for_task(task_description, context)
        
        if not tool:
            return ToolResult(
                tool_name="unknown",
                success=False,
                result=None,
                error="No appropriate tool found for task"
            )
        
        # Prepare invocation
        # Note: In a real implementation, we would use NLP to extract parameters
        # For now, we'll use context directly
        invocation = ToolInvocation(
            tool_name=tool.name,
            server=tool.server,
            arguments=context or {},
            context={"task": task_description}
        )
        
        # Invoke tool
        return self.invoke_tool(invocation)
    
    def get_tool_statistics(self) -> Dict[str, Any]:
        """Get statistics about tool usage."""
        total_invocations = len(self.invocation_history)
        successful_invocations = sum(1 for r in self.result_history if r.success)
        
        return {
            "total_tools": len(self.tools),
            "total_invocations": total_invocations,
            "successful_invocations": successful_invocations,
            "success_rate": successful_invocations / total_invocations if total_invocations > 0 else 0.0,
            "average_latency": sum(r.latency for r in self.result_history) / len(self.result_history) if self.result_history else 0.0,
            "tools_by_category": {
                category.value: len([t for t in self.tools.values() if t.category == category])
                for category in ToolCategory
            }
        }
    
    def integrate_with_cognitive_process(self, process_id: str, process_type: str, 
                                        callback: Callable[[ToolResult], None]):
        """
        Integrate MCP tools with a cognitive process.
        
        Args:
            process_id: ID of the cognitive process
            process_type: Type of cognitive process (reasoning, learning, etc.)
            callback: Callback function to handle tool results
        """
        logger.info(f"Integrating MCP tools with process: {process_id} ({process_type})")
        
        # Find relevant tools for this process type
        relevant_tools = []
        
        if process_type == "reasoning":
            # Reasoning processes might benefit from database queries and research papers
            relevant_tools.extend([t for t in self.tools.values() 
                                  if t.category in [ToolCategory.DATABASE, ToolCategory.SEARCH]])
        elif process_type == "learning":
            # Learning processes might benefit from model discovery and datasets
            relevant_tools.extend([t for t in self.tools.values() 
                                  if "model" in t.name or "dataset" in t.name])
        elif process_type == "perception":
            # Perception processes might benefit from model inference
            relevant_tools.extend([t for t in self.tools.values() 
                                  if t.category == ToolCategory.MODEL])
        
        logger.info(f"Found {len(relevant_tools)} relevant tools for {process_type} process")
        
        # Store integration for future reference
        # In a full implementation, this would set up event handlers and callbacks
        return {
            "process_id": process_id,
            "process_type": process_type,
            "available_tools": [t.name for t in relevant_tools],
            "integration_timestamp": datetime.now().isoformat()
        }


# ==================== Example Usage ====================

if __name__ == "__main__":
    # Initialize autonomous MCP agent
    agent = AutonomousMCPAgent()
    
    # List available tools
    print("\n=== Available Tools ===")
    for tool in agent.get_available_tools():
        print(f"- {tool.server}:{tool.name} ({tool.category.value}): {tool.description}")
    
    # Example: Search for models
    print("\n=== Example: Autonomous Task Execution ===")
    result = agent.autonomous_task_execution(
        "Search for reasoning models on Hugging Face",
        context={"query": "reasoning", "filter": "text-generation"}
    )
    print(f"Result: {result.to_dict()}")
    
    # Get statistics
    print("\n=== Tool Statistics ===")
    stats = agent.get_tool_statistics()
    for key, value in stats.items():
        print(f"{key}: {value}")
    
    # Example: Integration with cognitive process
    print("\n=== Example: Cognitive Process Integration ===")
    integration = agent.integrate_with_cognitive_process(
        process_id="reasoning-001",
        process_type="reasoning",
        callback=lambda result: print(f"Tool result received: {result.tool_name}")
    )
    print(f"Integration: {integration}")

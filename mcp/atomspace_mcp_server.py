#!/usr/bin/env python3
"""
AtomSpace MCP Server for OpenCog Collection
===========================================

This module implements a Model Context Protocol (MCP) server that provides
external tools and AI agents access to the OpenCog AtomSpace hypergraph.

Features:
- Query atoms by type, name, or attention value
- Create and link atoms
- Pattern matching and search
- Attention allocation
- Synergy metrics retrieval

MCP Protocol: https://modelcontextprotocol.io/

Author: OpenCog Collection Contributors
License: GPL-3.0+
"""

import asyncio
import json
import logging
from typing import Any, Dict, List, Optional
from datetime import datetime

# MCP server imports
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import (
    Tool,
    TextContent,
    ImageContent,
    EmbeddedResource,
    LoggingLevel
)

# Import our hypergraph persistence layer
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from database.hypergraph_persistence import HypergraphPersistence

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize MCP server
app = Server("atomspace-mcp-server")

# Initialize hypergraph persistence
persistence = HypergraphPersistence()


@app.list_tools()
async def list_tools() -> List[Tool]:
    """
    List all available MCP tools for AtomSpace interaction.
    """
    return [
        Tool(
            name="query_atoms",
            description="Query atoms from the AtomSpace hypergraph by type, attention value, or other criteria",
            inputSchema={
                "type": "object",
                "properties": {
                    "atom_type": {
                        "type": "string",
                        "description": "Filter by atom type (e.g., 'ConceptNode', 'PredicateNode')"
                    },
                    "min_attention": {
                        "type": "number",
                        "description": "Minimum attention value (0.0 to 1.0)"
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of results to return",
                        "default": 100
                    }
                }
            }
        ),
        Tool(
            name="create_atom",
            description="Create a new atom in the AtomSpace hypergraph",
            inputSchema={
                "type": "object",
                "properties": {
                    "atom_type": {
                        "type": "string",
                        "description": "Type of atom to create (e.g., 'ConceptNode', 'PredicateNode')",
                        "required": True
                    },
                    "name": {
                        "type": "string",
                        "description": "Name/identifier for the atom",
                        "required": True
                    },
                    "truth_value": {
                        "type": "number",
                        "description": "Truth value (0.0 to 1.0)",
                        "default": 1.0
                    },
                    "attention_value": {
                        "type": "number",
                        "description": "Initial attention value",
                        "default": 0.0
                    },
                    "metadata": {
                        "type": "object",
                        "description": "Additional metadata as JSON object"
                    }
                },
                "required": ["atom_type", "name"]
            }
        ),
        Tool(
            name="link_atoms",
            description="Create a link between two atoms in the hypergraph",
            inputSchema={
                "type": "object",
                "properties": {
                    "source_atom_id": {
                        "type": "string",
                        "description": "UUID of the source atom",
                        "required": True
                    },
                    "target_atom_id": {
                        "type": "string",
                        "description": "UUID of the target atom",
                        "required": True
                    },
                    "link_type": {
                        "type": "string",
                        "description": "Type of link (e.g., 'InheritanceLink', 'SimilarityLink')",
                        "required": True
                    },
                    "strength": {
                        "type": "number",
                        "description": "Link strength (0.0 to 1.0)",
                        "default": 1.0
                    }
                },
                "required": ["source_atom_id", "target_atom_id", "link_type"]
            }
        ),
        Tool(
            name="get_atom_neighbors",
            description="Get all atoms connected to a given atom via links",
            inputSchema={
                "type": "object",
                "properties": {
                    "atom_id": {
                        "type": "string",
                        "description": "UUID of the atom",
                        "required": True
                    },
                    "direction": {
                        "type": "string",
                        "description": "Link direction: 'incoming', 'outgoing', or 'both'",
                        "enum": ["incoming", "outgoing", "both"],
                        "default": "both"
                    }
                },
                "required": ["atom_id"]
            }
        ),
        Tool(
            name="allocate_attention",
            description="Allocate attention to an atom, increasing its priority in cognitive processing",
            inputSchema={
                "type": "object",
                "properties": {
                    "atom_id": {
                        "type": "string",
                        "description": "UUID of the atom",
                        "required": True
                    },
                    "attention_delta": {
                        "type": "number",
                        "description": "Change in attention value (positive or negative)",
                        "required": True
                    },
                    "reason": {
                        "type": "string",
                        "description": "Reason for attention allocation"
                    }
                },
                "required": ["atom_id", "attention_delta"]
            }
        ),
        Tool(
            name="get_synergy_metrics",
            description="Get cognitive synergy metrics and system performance data",
            inputSchema={
                "type": "object",
                "properties": {
                    "metric_name": {
                        "type": "string",
                        "description": "Filter by specific metric name"
                    },
                    "hours": {
                        "type": "integer",
                        "description": "Time window in hours",
                        "default": 24
                    }
                }
            }
        ),
        Tool(
            name="find_patterns",
            description="Search for discovered patterns in the cognitive architecture",
            inputSchema={
                "type": "object",
                "properties": {
                    "pattern_type": {
                        "type": "string",
                        "description": "Filter by pattern type"
                    },
                    "min_confidence": {
                        "type": "number",
                        "description": "Minimum confidence threshold (0.0 to 1.0)"
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of results",
                        "default": 50
                    }
                }
            }
        ),
        Tool(
            name="get_active_processes",
            description="Get information about active cognitive processes",
            inputSchema={
                "type": "object",
                "properties": {
                    "process_type": {
                        "type": "string",
                        "description": "Filter by process type"
                    },
                    "include_stuck": {
                        "type": "boolean",
                        "description": "Include stuck/bottlenecked processes",
                        "default": False
                    }
                }
            }
        )
    ]


@app.call_tool()
async def call_tool(name: str, arguments: Any) -> List[TextContent]:
    """
    Handle tool calls from MCP clients.
    """
    try:
        if name == "query_atoms":
            atoms = persistence.find_atoms(
                atom_type=arguments.get("atom_type"),
                min_attention=arguments.get("min_attention"),
                limit=arguments.get("limit", 100)
            )
            return [TextContent(
                type="text",
                text=json.dumps({
                    "success": True,
                    "count": len(atoms),
                    "atoms": atoms
                }, indent=2, default=str)
            )]
        
        elif name == "create_atom":
            atom_id = persistence.save_atom(
                atom_type=arguments["atom_type"],
                name=arguments["name"],
                truth_value=arguments.get("truth_value", 1.0),
                attention_value=arguments.get("attention_value", 0.0),
                metadata=arguments.get("metadata")
            )
            return [TextContent(
                type="text",
                text=json.dumps({
                    "success": True,
                    "atom_id": atom_id,
                    "message": f"Created atom: {arguments['atom_type']}:{arguments['name']}"
                }, indent=2)
            )]
        
        elif name == "link_atoms":
            link_id = persistence.create_link(
                source_atom_id=arguments["source_atom_id"],
                target_atom_id=arguments["target_atom_id"],
                link_type=arguments["link_type"],
                strength=arguments.get("strength", 1.0)
            )
            return [TextContent(
                type="text",
                text=json.dumps({
                    "success": True,
                    "link_id": link_id,
                    "message": f"Created link: {arguments['link_type']}"
                }, indent=2)
            )]
        
        elif name == "get_atom_neighbors":
            links = persistence.get_atom_links(
                atom_id=arguments["atom_id"],
                direction=arguments.get("direction", "both")
            )
            return [TextContent(
                type="text",
                text=json.dumps({
                    "success": True,
                    "count": len(links),
                    "links": links
                }, indent=2, default=str)
            )]
        
        elif name == "allocate_attention":
            persistence.update_attention(
                atom_id=arguments["atom_id"],
                attention_delta=arguments["attention_delta"],
                reason=arguments.get("reason")
            )
            return [TextContent(
                type="text",
                text=json.dumps({
                    "success": True,
                    "message": f"Allocated attention: {arguments['attention_delta']:+.3f}"
                }, indent=2)
            )]
        
        elif name == "get_synergy_metrics":
            metrics = persistence.get_metrics_summary(
                metric_name=arguments.get("metric_name"),
                hours=arguments.get("hours", 24)
            )
            return [TextContent(
                type="text",
                text=json.dumps({
                    "success": True,
                    "metrics": metrics
                }, indent=2, default=str)
            )]
        
        elif name == "find_patterns":
            # Query patterns from database
            query = persistence.client.table('patterns').select('*')
            
            if arguments.get("pattern_type"):
                query = query.eq('pattern_type', arguments["pattern_type"])
            
            if arguments.get("min_confidence"):
                query = query.gte('confidence', arguments["min_confidence"])
            
            result = query.order('frequency', desc=True).limit(
                arguments.get("limit", 50)
            ).execute()
            
            return [TextContent(
                type="text",
                text=json.dumps({
                    "success": True,
                    "count": len(result.data),
                    "patterns": result.data
                }, indent=2, default=str)
            )]
        
        elif name == "get_active_processes":
            query = persistence.client.table('cognitive_processes').select('*')
            
            if arguments.get("process_type"):
                query = query.eq('process_type', arguments["process_type"])
            
            if not arguments.get("include_stuck", False):
                query = query.eq('is_stuck', False)
            
            query = query.eq('status', 'active')
            result = query.order('priority', desc=True).execute()
            
            return [TextContent(
                type="text",
                text=json.dumps({
                    "success": True,
                    "count": len(result.data),
                    "processes": result.data
                }, indent=2, default=str)
            )]
        
        else:
            return [TextContent(
                type="text",
                text=json.dumps({
                    "success": False,
                    "error": f"Unknown tool: {name}"
                }, indent=2)
            )]
    
    except Exception as e:
        logger.error(f"Error executing tool {name}: {e}")
        return [TextContent(
            type="text",
            text=json.dumps({
                "success": False,
                "error": str(e)
            }, indent=2)
        )]


async def main():
    """
    Main entry point for the MCP server.
    """
    logger.info("Starting AtomSpace MCP Server...")
    
    async with stdio_server() as (read_stream, write_stream):
        await app.run(
            read_stream,
            write_stream,
            app.create_initialization_options()
        )


if __name__ == "__main__":
    asyncio.run(main())


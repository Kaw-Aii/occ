"""
Hypergraph Persistence Layer for OpenCog Collection
===================================================

This module provides persistent storage for hypergraph-based cognitive patterns
using Supabase PostgreSQL. It enables:

1. Persistent pattern storage across sessions
2. Efficient hypergraph query operations
3. Cognitive state snapshots and versioning
4. Distributed cognitive architectures
5. Research reproducibility

Author: OpenCog Collection Contributors
License: GPL-3.0+
"""

import os
import json
import logging
from typing import Dict, List, Any, Optional, Set, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
from supabase import create_client, Client
import uuid

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class HypergraphNode:
    """Represents a node in the hypergraph."""
    node_id: str
    node_type: str
    content: Any
    attention_value: float = 0.0
    timestamp: datetime = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()
        if self.metadata is None:
            self.metadata = {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "node_id": self.node_id,
            "node_type": self.node_type,
            "content": json.dumps(self.content) if not isinstance(self.content, str) else self.content,
            "attention_value": self.attention_value,
            "timestamp": self.timestamp.isoformat(),
            "metadata": json.dumps(self.metadata)
        }


@dataclass
class HypergraphEdge:
    """Represents an edge (relation) in the hypergraph."""
    edge_id: str
    edge_type: str
    source_nodes: List[str]
    target_nodes: List[str]
    weight: float = 1.0
    confidence: float = 1.0
    timestamp: datetime = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()
        if self.metadata is None:
            self.metadata = {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "edge_id": self.edge_id,
            "edge_type": self.edge_type,
            "source_nodes": json.dumps(self.source_nodes),
            "target_nodes": json.dumps(self.target_nodes),
            "weight": self.weight,
            "confidence": self.confidence,
            "timestamp": self.timestamp.isoformat(),
            "metadata": json.dumps(self.metadata)
        }


@dataclass
class CognitivePattern:
    """Represents a cognitive pattern stored in the hypergraph."""
    pattern_id: str
    source_process: str
    pattern_type: str
    content: Any
    confidence: float
    attention_value: float
    hypergraph_nodes: List[str]
    hypergraph_edges: List[str]
    consumers: Set[str]
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "pattern_id": self.pattern_id,
            "source_process": self.source_process,
            "pattern_type": self.pattern_type,
            "content": json.dumps(self.content) if not isinstance(self.content, str) else self.content,
            "confidence": self.confidence,
            "attention_value": self.attention_value,
            "hypergraph_nodes": json.dumps(self.hypergraph_nodes),
            "hypergraph_edges": json.dumps(self.hypergraph_edges),
            "consumers": json.dumps(list(self.consumers)),
            "timestamp": self.timestamp.isoformat()
        }


@dataclass
class CognitiveSnapshot:
    """Represents a snapshot of the entire cognitive state."""
    snapshot_id: str
    timestamp: datetime
    processes: Dict[str, Any]
    metrics: Dict[str, Any]
    hypergraph_state: Dict[str, Any]
    description: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "snapshot_id": self.snapshot_id,
            "timestamp": self.timestamp.isoformat(),
            "processes": json.dumps(self.processes),
            "metrics": json.dumps(self.metrics),
            "hypergraph_state": json.dumps(self.hypergraph_state),
            "description": self.description
        }


class HypergraphPersistenceLayer:
    """
    Manages persistent storage of hypergraph-based cognitive patterns.
    
    Uses Supabase PostgreSQL for efficient storage and querying of:
    - Hypergraph nodes and edges
    - Cognitive patterns
    - Cognitive state snapshots
    """
    
    def __init__(self, supabase_url: Optional[str] = None, supabase_key: Optional[str] = None):
        """
        Initialize the persistence layer.
        
        Args:
            supabase_url: Supabase project URL (defaults to SUPABASE_URL env var)
            supabase_key: Supabase API key (defaults to SUPABASE_KEY env var)
        """
        self.supabase_url = supabase_url or os.getenv("SUPABASE_URL")
        self.supabase_key = supabase_key or os.getenv("SUPABASE_KEY")
        
        if not self.supabase_url or not self.supabase_key:
            logger.warning("Supabase credentials not provided. Persistence layer will operate in mock mode.")
            self.client = None
            self.mock_mode = True
            self.mock_storage = {
                "nodes": {},
                "edges": {},
                "patterns": {},
                "snapshots": {}
            }
        else:
            try:
                self.client: Client = create_client(self.supabase_url, self.supabase_key)
                self.mock_mode = False
                self._initialize_tables()
                logger.info("Hypergraph Persistence Layer initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize Supabase client: {e}")
                self.client = None
                self.mock_mode = True
                self.mock_storage = {
                    "nodes": {},
                    "edges": {},
                    "patterns": {},
                    "snapshots": {}
                }
    
    def _initialize_tables(self):
        """Initialize database tables if they don't exist."""
        if self.mock_mode:
            return
        
        # Note: In production, tables should be created via Supabase migrations
        # This is a placeholder for table initialization logic
        logger.info("Database tables should be created via Supabase migrations")
    
    # ==================== Node Operations ====================
    
    def store_node(self, node: HypergraphNode) -> bool:
        """Store a hypergraph node."""
        try:
            if self.mock_mode:
                self.mock_storage["nodes"][node.node_id] = node
                logger.info(f"[MOCK] Stored node: {node.node_id}")
                return True
            
            data = node.to_dict()
            self.client.table("hypergraph_nodes").upsert(data).execute()
            logger.info(f"Stored node: {node.node_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to store node {node.node_id}: {e}")
            return False
    
    def get_node(self, node_id: str) -> Optional[HypergraphNode]:
        """Retrieve a hypergraph node by ID."""
        try:
            if self.mock_mode:
                return self.mock_storage["nodes"].get(node_id)
            
            response = self.client.table("hypergraph_nodes").select("*").eq("node_id", node_id).execute()
            if response.data:
                data = response.data[0]
                return HypergraphNode(
                    node_id=data["node_id"],
                    node_type=data["node_type"],
                    content=json.loads(data["content"]) if data["content"] else None,
                    attention_value=data["attention_value"],
                    timestamp=datetime.fromisoformat(data["timestamp"]),
                    metadata=json.loads(data["metadata"]) if data["metadata"] else {}
                )
            return None
        except Exception as e:
            logger.error(f"Failed to retrieve node {node_id}: {e}")
            return None
    
    def query_nodes_by_type(self, node_type: str, limit: int = 100) -> List[HypergraphNode]:
        """Query nodes by type."""
        try:
            if self.mock_mode:
                nodes = [n for n in self.mock_storage["nodes"].values() if n.node_type == node_type]
                result = nodes[:limit]
                logger.info(f"[MOCK] Queried {len(result)} nodes of type {node_type}")
                return result
            
            response = self.client.table("hypergraph_nodes").select("*").eq("node_type", node_type).limit(limit).execute()
            nodes = []
            for data in response.data:
                nodes.append(HypergraphNode(
                    node_id=data["node_id"],
                    node_type=data["node_type"],
                    content=json.loads(data["content"]) if data["content"] else None,
                    attention_value=data["attention_value"],
                    timestamp=datetime.fromisoformat(data["timestamp"]),
                    metadata=json.loads(data["metadata"]) if data["metadata"] else {}
                ))
            return nodes
        except Exception as e:
            logger.error(f"Failed to query nodes by type {node_type}: {e}")
            return []
    
    # ==================== Edge Operations ====================
    
    def store_edge(self, edge: HypergraphEdge) -> bool:
        """Store a hypergraph edge."""
        try:
            if self.mock_mode:
                self.mock_storage["edges"][edge.edge_id] = edge
                logger.info(f"[MOCK] Stored edge: {edge.edge_id}")
                return True
            
            data = edge.to_dict()
            self.client.table("hypergraph_edges").upsert(data).execute()
            logger.info(f"Stored edge: {edge.edge_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to store edge {edge.edge_id}: {e}")
            return False
    
    def get_edge(self, edge_id: str) -> Optional[HypergraphEdge]:
        """Retrieve a hypergraph edge by ID."""
        try:
            if self.mock_mode:
                return self.mock_storage["edges"].get(edge_id)
            
            response = self.client.table("hypergraph_edges").select("*").eq("edge_id", edge_id).execute()
            if response.data:
                data = response.data[0]
                return HypergraphEdge(
                    edge_id=data["edge_id"],
                    edge_type=data["edge_type"],
                    source_nodes=json.loads(data["source_nodes"]),
                    target_nodes=json.loads(data["target_nodes"]),
                    weight=data["weight"],
                    confidence=data["confidence"],
                    timestamp=datetime.fromisoformat(data["timestamp"]),
                    metadata=json.loads(data["metadata"]) if data["metadata"] else {}
                )
            return None
        except Exception as e:
            logger.error(f"Failed to retrieve edge {edge_id}: {e}")
            return None
    
    def query_edges_by_node(self, node_id: str) -> List[HypergraphEdge]:
        """Query all edges connected to a node."""
        try:
            if self.mock_mode:
                edges = [e for e in self.mock_storage["edges"].values() 
                        if node_id in e.source_nodes or node_id in e.target_nodes]
                return edges
            
            # This requires a more complex query in production
            # For now, we'll use a simple approach
            response = self.client.table("hypergraph_edges").select("*").execute()
            edges = []
            for data in response.data:
                source_nodes = json.loads(data["source_nodes"])
                target_nodes = json.loads(data["target_nodes"])
                if node_id in source_nodes or node_id in target_nodes:
                    edges.append(HypergraphEdge(
                        edge_id=data["edge_id"],
                        edge_type=data["edge_type"],
                        source_nodes=source_nodes,
                        target_nodes=target_nodes,
                        weight=data["weight"],
                        confidence=data["confidence"],
                        timestamp=datetime.fromisoformat(data["timestamp"]),
                        metadata=json.loads(data["metadata"]) if data["metadata"] else {}
                    ))
            return edges
        except Exception as e:
            logger.error(f"Failed to query edges for node {node_id}: {e}")
            return []
    
    # ==================== Pattern Operations ====================
    
    def store_pattern(self, pattern: CognitivePattern) -> bool:
        """Store a cognitive pattern."""
        try:
            if self.mock_mode:
                self.mock_storage["patterns"][pattern.pattern_id] = pattern
                logger.info(f"[MOCK] Stored pattern: {pattern.pattern_id}")
                return True
            
            data = pattern.to_dict()
            self.client.table("cognitive_patterns").upsert(data).execute()
            logger.info(f"Stored pattern: {pattern.pattern_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to store pattern {pattern.pattern_id}: {e}")
            return False
    
    def get_pattern(self, pattern_id: str) -> Optional[CognitivePattern]:
        """Retrieve a cognitive pattern by ID."""
        try:
            if self.mock_mode:
                return self.mock_storage["patterns"].get(pattern_id)
            
            response = self.client.table("cognitive_patterns").select("*").eq("pattern_id", pattern_id).execute()
            if response.data:
                data = response.data[0]
                return CognitivePattern(
                    pattern_id=data["pattern_id"],
                    source_process=data["source_process"],
                    pattern_type=data["pattern_type"],
                    content=json.loads(data["content"]) if data["content"] else None,
                    confidence=data["confidence"],
                    attention_value=data["attention_value"],
                    hypergraph_nodes=json.loads(data["hypergraph_nodes"]),
                    hypergraph_edges=json.loads(data["hypergraph_edges"]),
                    consumers=set(json.loads(data["consumers"])),
                    timestamp=datetime.fromisoformat(data["timestamp"])
                )
            return None
        except Exception as e:
            logger.error(f"Failed to retrieve pattern {pattern_id}: {e}")
            return None
    
    def query_patterns_by_type(self, pattern_type: str, limit: int = 100) -> List[CognitivePattern]:
        """Query patterns by type."""
        try:
            if self.mock_mode:
                patterns = [p for p in self.mock_storage["patterns"].values() if p.pattern_type == pattern_type]
                result = patterns[:limit]
                logger.info(f"[MOCK] Queried {len(result)} patterns of type {pattern_type}")
                return result
            
            response = self.client.table("cognitive_patterns").select("*").eq("pattern_type", pattern_type).limit(limit).execute()
            patterns = []
            for data in response.data:
                patterns.append(CognitivePattern(
                    pattern_id=data["pattern_id"],
                    source_process=data["source_process"],
                    pattern_type=data["pattern_type"],
                    content=json.loads(data["content"]) if data["content"] else None,
                    confidence=data["confidence"],
                    attention_value=data["attention_value"],
                    hypergraph_nodes=json.loads(data["hypergraph_nodes"]),
                    hypergraph_edges=json.loads(data["hypergraph_edges"]),
                    consumers=set(json.loads(data["consumers"])),
                    timestamp=datetime.fromisoformat(data["timestamp"])
                ))
            return patterns
        except Exception as e:
            logger.error(f"Failed to query patterns by type {pattern_type}: {e}")
            return []
    
    # ==================== Snapshot Operations ====================
    
    def create_snapshot(self, snapshot: CognitiveSnapshot) -> bool:
        """Create a cognitive state snapshot."""
        try:
            if self.mock_mode:
                self.mock_storage["snapshots"][snapshot.snapshot_id] = snapshot
                logger.info(f"[MOCK] Created snapshot: {snapshot.snapshot_id}")
                return True
            
            data = snapshot.to_dict()
            self.client.table("cognitive_snapshots").insert(data).execute()
            logger.info(f"Created snapshot: {snapshot.snapshot_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to create snapshot {snapshot.snapshot_id}: {e}")
            return False
    
    def get_snapshot(self, snapshot_id: str) -> Optional[CognitiveSnapshot]:
        """Retrieve a cognitive snapshot by ID."""
        try:
            if self.mock_mode:
                return self.mock_storage["snapshots"].get(snapshot_id)
            
            response = self.client.table("cognitive_snapshots").select("*").eq("snapshot_id", snapshot_id).execute()
            if response.data:
                data = response.data[0]
                return CognitiveSnapshot(
                    snapshot_id=data["snapshot_id"],
                    timestamp=datetime.fromisoformat(data["timestamp"]),
                    processes=json.loads(data["processes"]),
                    metrics=json.loads(data["metrics"]),
                    hypergraph_state=json.loads(data["hypergraph_state"]),
                    description=data["description"]
                )
            return None
        except Exception as e:
            logger.error(f"Failed to retrieve snapshot {snapshot_id}: {e}")
            return None
    
    def list_snapshots(self, limit: int = 10) -> List[CognitiveSnapshot]:
        """List recent cognitive snapshots."""
        try:
            if self.mock_mode:
                snapshots = list(self.mock_storage["snapshots"].values())
                snapshots.sort(key=lambda s: s.timestamp, reverse=True)
                return snapshots[:limit]
            
            response = self.client.table("cognitive_snapshots").select("*").order("timestamp", desc=True).limit(limit).execute()
            snapshots = []
            for data in response.data:
                snapshots.append(CognitiveSnapshot(
                    snapshot_id=data["snapshot_id"],
                    timestamp=datetime.fromisoformat(data["timestamp"]),
                    processes=json.loads(data["processes"]),
                    metrics=json.loads(data["metrics"]),
                    hypergraph_state=json.loads(data["hypergraph_state"]),
                    description=data["description"]
                ))
            return snapshots
        except Exception as e:
            logger.error(f"Failed to list snapshots: {e}")
            return []
    
    # ==================== Utility Methods ====================
    
    def get_statistics(self) -> Dict[str, int]:
        """Get storage statistics."""
        try:
            if self.mock_mode:
                stats = {
                    "nodes": len(self.mock_storage["nodes"]),
                    "edges": len(self.mock_storage["edges"]),
                    "patterns": len(self.mock_storage["patterns"]),
                    "snapshots": len(self.mock_storage["snapshots"])
                }
                logger.info(f"[MOCK] Storage statistics: {stats}")
                return stats
            
            stats = {}
            for table in ["hypergraph_nodes", "hypergraph_edges", "cognitive_patterns", "cognitive_snapshots"]:
                response = self.client.table(table).select("*", count="exact").execute()
                stats[table] = response.count if hasattr(response, 'count') else len(response.data)
            return stats
        except Exception as e:
            logger.error(f"Failed to get statistics: {e}")
            return {}


# ==================== Example Usage ====================

if __name__ == "__main__":
    # Initialize persistence layer
    persistence = HypergraphPersistenceLayer()
    
    # Create and store a node
    node = HypergraphNode(
        node_id=str(uuid.uuid4()),
        node_type="concept",
        content={"name": "cognitive_synergy", "definition": "Emergent intelligence from module interaction"},
        attention_value=0.9
    )
    persistence.store_node(node)
    
    # Create and store an edge
    edge = HypergraphEdge(
        edge_id=str(uuid.uuid4()),
        edge_type="enables",
        source_nodes=[node.node_id],
        target_nodes=["emergent_intelligence"],
        weight=0.95,
        confidence=0.9
    )
    persistence.store_edge(edge)
    
    # Create and store a pattern
    pattern = CognitivePattern(
        pattern_id=str(uuid.uuid4()),
        source_process="reasoning",
        pattern_type="inference",
        content={"rule": "If A and B then C", "confidence": 0.85},
        confidence=0.85,
        attention_value=0.8,
        hypergraph_nodes=[node.node_id],
        hypergraph_edges=[edge.edge_id],
        consumers={"learning", "planning"}
    )
    persistence.store_pattern(pattern)
    
    # Create a snapshot
    snapshot = CognitiveSnapshot(
        snapshot_id=str(uuid.uuid4()),
        timestamp=datetime.now(),
        processes={"reasoning": "active", "learning": "active"},
        metrics={"attention_efficiency": 0.87, "cross_module_patterns": 42},
        hypergraph_state={"nodes": 150, "edges": 320},
        description="Initial cognitive state after pattern discovery"
    )
    persistence.create_snapshot(snapshot)
    
    # Get statistics
    stats = persistence.get_statistics()
    print(f"Storage statistics: {stats}")

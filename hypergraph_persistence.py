"""
Hypergraph Persistence Layer
============================

This module provides persistent storage for hypergraph cognitive structures
using both Neon (PostgreSQL) and Supabase for distributed cognition.

Features:
- Persistent atom and link storage
- Temporal versioning of cognitive state
- Efficient pattern matching queries
- Real-time synchronization via Supabase
- Cross-session cognitive continuity

Author: OpenCog Collection Contributors
License: GPL-3.0+
"""

import os
import logging
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, asdict
from datetime import datetime
import json
import hashlib

# Database clients
try:
    from supabase import create_client, Client
    SUPABASE_AVAILABLE = True
except ImportError:
    SUPABASE_AVAILABLE = False
    logging.warning("Supabase not available. Install with: pip install supabase")

try:
    import psycopg2
    from psycopg2.extras import Json, RealDictCursor
    PSYCOPG2_AVAILABLE = True
except ImportError:
    PSYCOPG2_AVAILABLE = False
    logging.warning("psycopg2 not available. Install with: pip install psycopg2-binary")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class PersistedAtom:
    """Atom representation for database storage."""
    atom_id: str
    atom_type: str
    name: str
    truth_strength: float = 1.0
    truth_confidence: float = 1.0
    attention_value: float = 0.0
    metadata: Dict[str, Any] = None
    created_at: datetime = None
    updated_at: datetime = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
        if self.created_at is None:
            self.created_at = datetime.now()
        if self.updated_at is None:
            self.updated_at = datetime.now()


@dataclass
class PersistedLink:
    """Link representation for database storage."""
    link_id: str
    link_type: str
    outgoing: List[str]  # atom_ids
    truth_strength: float = 1.0
    truth_confidence: float = 1.0
    attention_value: float = 0.0
    metadata: Dict[str, Any] = None
    created_at: datetime = None
    updated_at: datetime = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
        if self.created_at is None:
            self.created_at = datetime.now()
        if self.updated_at is None:
            self.updated_at = datetime.now()


@dataclass
class CognitiveSnapshot:
    """Snapshot of cognitive state at a point in time."""
    snapshot_id: str
    timestamp: datetime
    atom_count: int
    link_count: int
    total_attention: float
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class NeonHypergraphStore:
    """
    Neon PostgreSQL backend for hypergraph storage.
    Provides high-performance persistent storage with SQL queries.
    """
    
    def __init__(self, connection_string: Optional[str] = None):
        """Initialize Neon connection."""
        if not PSYCOPG2_AVAILABLE:
            raise ImportError("psycopg2 is required for NeonHypergraphStore")
        
        self.connection_string = connection_string or self._get_connection_string()
        self.conn = None
        self._connect()
        self._ensure_schema()
        
        logger.info("NeonHypergraphStore initialized")
    
    def _get_connection_string(self) -> str:
        """Get connection string from environment or MCP."""
        # Try environment variable first
        conn_str = os.getenv("NEON_DATABASE_URL")
        if conn_str:
            return conn_str
        
        # TODO: Get from MCP if not in environment
        raise ValueError("NEON_DATABASE_URL not set. Please configure Neon connection.")
    
    def _connect(self):
        """Establish database connection."""
        try:
            self.conn = psycopg2.connect(self.connection_string)
            self.conn.autocommit = False
            logger.info("Connected to Neon database")
        except Exception as e:
            logger.error(f"Failed to connect to Neon: {e}")
            raise
    
    def _ensure_schema(self):
        """Create database schema if it doesn't exist."""
        schema_sql = """
        -- Atoms table
        CREATE TABLE IF NOT EXISTS atoms (
            atom_id VARCHAR(255) PRIMARY KEY,
            atom_type VARCHAR(100) NOT NULL,
            name TEXT NOT NULL,
            truth_strength FLOAT DEFAULT 1.0,
            truth_confidence FLOAT DEFAULT 1.0,
            attention_value FLOAT DEFAULT 0.0,
            metadata JSONB DEFAULT '{}',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        
        -- Links table
        CREATE TABLE IF NOT EXISTS links (
            link_id VARCHAR(255) PRIMARY KEY,
            link_type VARCHAR(100) NOT NULL,
            outgoing TEXT[] NOT NULL,
            truth_strength FLOAT DEFAULT 1.0,
            truth_confidence FLOAT DEFAULT 1.0,
            attention_value FLOAT DEFAULT 0.0,
            metadata JSONB DEFAULT '{}',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        
        -- Cognitive snapshots table
        CREATE TABLE IF NOT EXISTS cognitive_snapshots (
            snapshot_id VARCHAR(255) PRIMARY KEY,
            timestamp TIMESTAMP NOT NULL,
            atom_count INTEGER NOT NULL,
            link_count INTEGER NOT NULL,
            total_attention FLOAT NOT NULL,
            metadata JSONB DEFAULT '{}',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        
        -- Indices for performance
        CREATE INDEX IF NOT EXISTS idx_atoms_type ON atoms(atom_type);
        CREATE INDEX IF NOT EXISTS idx_atoms_attention ON atoms(attention_value DESC);
        CREATE INDEX IF NOT EXISTS idx_atoms_name ON atoms(name);
        CREATE INDEX IF NOT EXISTS idx_links_type ON links(link_type);
        CREATE INDEX IF NOT EXISTS idx_links_attention ON links(attention_value DESC);
        CREATE INDEX IF NOT EXISTS idx_links_outgoing ON links USING GIN(outgoing);
        CREATE INDEX IF NOT EXISTS idx_snapshots_timestamp ON cognitive_snapshots(timestamp DESC);
        """
        
        try:
            with self.conn.cursor() as cur:
                cur.execute(schema_sql)
            self.conn.commit()
            logger.info("Database schema ensured")
        except Exception as e:
            self.conn.rollback()
            logger.error(f"Failed to create schema: {e}")
            raise
    
    def store_atom(self, atom: PersistedAtom) -> bool:
        """Store or update an atom."""
        sql = """
        INSERT INTO atoms (atom_id, atom_type, name, truth_strength, truth_confidence, 
                          attention_value, metadata, created_at, updated_at)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (atom_id) DO UPDATE SET
            atom_type = EXCLUDED.atom_type,
            name = EXCLUDED.name,
            truth_strength = EXCLUDED.truth_strength,
            truth_confidence = EXCLUDED.truth_confidence,
            attention_value = EXCLUDED.attention_value,
            metadata = EXCLUDED.metadata,
            updated_at = EXCLUDED.updated_at
        """
        
        try:
            with self.conn.cursor() as cur:
                cur.execute(sql, (
                    atom.atom_id, atom.atom_type, atom.name,
                    atom.truth_strength, atom.truth_confidence,
                    atom.attention_value, Json(atom.metadata),
                    atom.created_at, atom.updated_at
                ))
            self.conn.commit()
            logger.debug(f"Stored atom: {atom.atom_id}")
            return True
        except Exception as e:
            self.conn.rollback()
            logger.error(f"Failed to store atom {atom.atom_id}: {e}")
            return False
    
    def store_link(self, link: PersistedLink) -> bool:
        """Store or update a link."""
        sql = """
        INSERT INTO links (link_id, link_type, outgoing, truth_strength, truth_confidence,
                          attention_value, metadata, created_at, updated_at)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (link_id) DO UPDATE SET
            link_type = EXCLUDED.link_type,
            outgoing = EXCLUDED.outgoing,
            truth_strength = EXCLUDED.truth_strength,
            truth_confidence = EXCLUDED.truth_confidence,
            attention_value = EXCLUDED.attention_value,
            metadata = EXCLUDED.metadata,
            updated_at = EXCLUDED.updated_at
        """
        
        try:
            with self.conn.cursor() as cur:
                cur.execute(sql, (
                    link.link_id, link.link_type, link.outgoing,
                    link.truth_strength, link.truth_confidence,
                    link.attention_value, Json(link.metadata),
                    link.created_at, link.updated_at
                ))
            self.conn.commit()
            logger.debug(f"Stored link: {link.link_id}")
            return True
        except Exception as e:
            self.conn.rollback()
            logger.error(f"Failed to store link {link.link_id}: {e}")
            return False
    
    def get_atom(self, atom_id: str) -> Optional[PersistedAtom]:
        """Retrieve an atom by ID."""
        sql = "SELECT * FROM atoms WHERE atom_id = %s"
        
        try:
            with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(sql, (atom_id,))
                row = cur.fetchone()
                if row:
                    return PersistedAtom(**dict(row))
                return None
        except Exception as e:
            logger.error(f"Failed to get atom {atom_id}: {e}")
            return None
    
    def get_atoms_by_type(self, atom_type: str, limit: int = 100) -> List[PersistedAtom]:
        """Retrieve atoms by type."""
        sql = "SELECT * FROM atoms WHERE atom_type = %s ORDER BY attention_value DESC LIMIT %s"
        
        try:
            with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(sql, (atom_type, limit))
                rows = cur.fetchall()
                return [PersistedAtom(**dict(row)) for row in rows]
        except Exception as e:
            logger.error(f"Failed to get atoms by type {atom_type}: {e}")
            return []
    
    def get_high_attention_atoms(self, threshold: float = 0.5, limit: int = 100) -> List[PersistedAtom]:
        """Retrieve atoms with high attention values."""
        sql = """
        SELECT * FROM atoms 
        WHERE attention_value >= %s 
        ORDER BY attention_value DESC 
        LIMIT %s
        """
        
        try:
            with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(sql, (threshold, limit))
                rows = cur.fetchall()
                return [PersistedAtom(**dict(row)) for row in rows]
        except Exception as e:
            logger.error(f"Failed to get high attention atoms: {e}")
            return []
    
    def get_incoming_links(self, atom_id: str) -> List[PersistedLink]:
        """Get all links that include this atom in their outgoing set."""
        sql = "SELECT * FROM links WHERE %s = ANY(outgoing)"
        
        try:
            with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(sql, (atom_id,))
                rows = cur.fetchall()
                return [PersistedLink(**dict(row)) for row in rows]
        except Exception as e:
            logger.error(f"Failed to get incoming links for {atom_id}: {e}")
            return []
    
    def create_snapshot(self, snapshot: CognitiveSnapshot) -> bool:
        """Create a cognitive state snapshot."""
        sql = """
        INSERT INTO cognitive_snapshots (snapshot_id, timestamp, atom_count, link_count,
                                        total_attention, metadata)
        VALUES (%s, %s, %s, %s, %s, %s)
        """
        
        try:
            with self.conn.cursor() as cur:
                cur.execute(sql, (
                    snapshot.snapshot_id, snapshot.timestamp,
                    snapshot.atom_count, snapshot.link_count,
                    snapshot.total_attention, Json(snapshot.metadata)
                ))
            self.conn.commit()
            logger.info(f"Created snapshot: {snapshot.snapshot_id}")
            return True
        except Exception as e:
            self.conn.rollback()
            logger.error(f"Failed to create snapshot: {e}")
            return False
    
    def close(self):
        """Close database connection."""
        if self.conn:
            self.conn.close()
            logger.info("Neon connection closed")


class SupabaseHypergraphStore:
    """
    Supabase backend for real-time hypergraph synchronization.
    Provides real-time updates and distributed cognition capabilities.
    """
    
    def __init__(self, url: Optional[str] = None, key: Optional[str] = None):
        """Initialize Supabase client."""
        if not SUPABASE_AVAILABLE:
            raise ImportError("Supabase is required for SupabaseHypergraphStore")
        
        self.url = url or os.getenv("SUPABASE_URL")
        self.key = key or os.getenv("SUPABASE_KEY")
        
        if not self.url or not self.key:
            raise ValueError("SUPABASE_URL and SUPABASE_KEY must be set")
        
        self.client: Client = create_client(self.url, self.key)
        logger.info("SupabaseHypergraphStore initialized")
    
    def store_atom(self, atom: PersistedAtom) -> bool:
        """Store or update an atom in Supabase."""
        try:
            data = {
                "atom_id": atom.atom_id,
                "atom_type": atom.atom_type,
                "name": atom.name,
                "truth_strength": atom.truth_strength,
                "truth_confidence": atom.truth_confidence,
                "attention_value": atom.attention_value,
                "metadata": atom.metadata,
                "updated_at": atom.updated_at.isoformat()
            }
            
            result = self.client.table("atoms").upsert(data).execute()
            logger.debug(f"Stored atom in Supabase: {atom.atom_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to store atom in Supabase: {e}")
            return False
    
    def get_atom(self, atom_id: str) -> Optional[PersistedAtom]:
        """Retrieve an atom from Supabase."""
        try:
            result = self.client.table("atoms").select("*").eq("atom_id", atom_id).execute()
            if result.data:
                data = result.data[0]
                return PersistedAtom(
                    atom_id=data["atom_id"],
                    atom_type=data["atom_type"],
                    name=data["name"],
                    truth_strength=data["truth_strength"],
                    truth_confidence=data["truth_confidence"],
                    attention_value=data["attention_value"],
                    metadata=data.get("metadata", {}),
                    created_at=datetime.fromisoformat(data["created_at"]),
                    updated_at=datetime.fromisoformat(data["updated_at"])
                )
            return None
        except Exception as e:
            logger.error(f"Failed to get atom from Supabase: {e}")
            return None


class HypergraphPersistenceManager:
    """
    Unified persistence manager that coordinates between Neon and Supabase.
    Provides high-level API for cognitive state persistence.
    """
    
    def __init__(self, use_neon: bool = True, use_supabase: bool = True):
        """Initialize persistence manager."""
        self.neon_store = None
        self.supabase_store = None
        
        if use_neon and PSYCOPG2_AVAILABLE:
            try:
                self.neon_store = NeonHypergraphStore()
                logger.info("Neon store enabled")
            except Exception as e:
                logger.warning(f"Failed to initialize Neon store: {e}")
        
        if use_supabase and SUPABASE_AVAILABLE:
            try:
                self.supabase_store = SupabaseHypergraphStore()
                logger.info("Supabase store enabled")
            except Exception as e:
                logger.warning(f"Failed to initialize Supabase store: {e}")
        
        if not self.neon_store and not self.supabase_store:
            logger.warning("No persistence backends available")
    
    def store_atom(self, atom: PersistedAtom) -> bool:
        """Store atom to all available backends."""
        success = False
        
        if self.neon_store:
            success = self.neon_store.store_atom(atom) or success
        
        if self.supabase_store:
            success = self.supabase_store.store_atom(atom) or success
        
        return success
    
    def get_atom(self, atom_id: str) -> Optional[PersistedAtom]:
        """Retrieve atom from available backends (Neon preferred)."""
        if self.neon_store:
            return self.neon_store.get_atom(atom_id)
        
        if self.supabase_store:
            return self.supabase_store.get_atom(atom_id)
        
        return None
    
    def close(self):
        """Close all connections."""
        if self.neon_store:
            self.neon_store.close()
        
        logger.info("Persistence manager closed")


# Example usage
if __name__ == "__main__":
    # Initialize persistence manager
    manager = HypergraphPersistenceManager()
    
    # Create and store an atom
    atom = PersistedAtom(
        atom_id="concept_001",
        atom_type="concept",
        name="cognitive_synergy",
        truth_strength=0.95,
        truth_confidence=0.85,
        attention_value=0.75,
        metadata={"source": "research_paper", "domain": "agi"}
    )
    
    manager.store_atom(atom)
    
    # Retrieve the atom
    retrieved = manager.get_atom("concept_001")
    if retrieved:
        print(f"Retrieved atom: {retrieved.name}")
    
    manager.close()


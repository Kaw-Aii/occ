#!/usr/bin/env python3
"""
Hypergraph Persistence Layer for OpenCog Collection
===================================================

This module provides database persistence for the hypergraph memory structure,
enabling cognitive synergy data to be stored and retrieved from Supabase/Neon.

Features:
- Atom storage and retrieval
- Link management
- Pattern persistence
- Synergy event logging
- Metrics tracking

Author: OpenCog Collection Contributors
License: GPL-3.0+
"""

import os
import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from supabase import create_client, Client
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HypergraphPersistence:
    """
    Database persistence layer for hypergraph memory and cognitive synergy.
    """
    
    def __init__(self, supabase_url: Optional[str] = None, supabase_key: Optional[str] = None):
        """
        Initialize database connection.
        
        Args:
            supabase_url: Supabase project URL (defaults to SUPABASE_URL env var)
            supabase_key: Supabase API key (defaults to SUPABASE_KEY env var)
        """
        self.url = supabase_url or os.environ.get('SUPABASE_URL')
        self.key = supabase_key or os.environ.get('SUPABASE_KEY')
        
        if not self.url or not self.key:
            raise ValueError("Supabase URL and key must be provided or set in environment variables")
        
        self.client: Client = create_client(self.url, self.key)
        logger.info("Hypergraph persistence layer initialized")
    
    # Atom operations
    
    def save_atom(self, atom_type: str, name: str, truth_value: float = 1.0,
                  attention_value: float = 0.0, metadata: Dict[str, Any] = None) -> str:
        """
        Save an atom to the database.
        
        Args:
            atom_type: Type of the atom (e.g., 'ConceptNode', 'PredicateNode')
            name: Name/identifier of the atom
            truth_value: Truth value (0.0 to 1.0)
            attention_value: Attention value
            metadata: Additional metadata as JSON
        
        Returns:
            atom_id: UUID of the created/updated atom
        """
        try:
            data = {
                'atom_type': atom_type,
                'name': name,
                'truth_value': truth_value,
                'attention_value': attention_value,
                'metadata': metadata or {}
            }
            
            # Upsert: insert or update if exists
            result = self.client.table('atoms').upsert(data, on_conflict='atom_type,name').execute()
            
            if result.data:
                atom_id = result.data[0]['atom_id']
                logger.debug(f"Saved atom: {atom_type}:{name} (ID: {atom_id})")
                return atom_id
            else:
                raise Exception("Failed to save atom")
                
        except Exception as e:
            logger.error(f"Error saving atom: {e}")
            raise
    
    def get_atom(self, atom_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve an atom by ID.
        
        Args:
            atom_id: UUID of the atom
        
        Returns:
            Atom data as dictionary or None if not found
        """
        try:
            result = self.client.table('atoms').select('*').eq('atom_id', atom_id).execute()
            return result.data[0] if result.data else None
        except Exception as e:
            logger.error(f"Error retrieving atom: {e}")
            return None
    
    def find_atoms(self, atom_type: Optional[str] = None, 
                   min_attention: Optional[float] = None,
                   limit: int = 100) -> List[Dict[str, Any]]:
        """
        Find atoms matching criteria.
        
        Args:
            atom_type: Filter by atom type
            min_attention: Minimum attention value
            limit: Maximum number of results
        
        Returns:
            List of atom dictionaries
        """
        try:
            query = self.client.table('atoms').select('*')
            
            if atom_type:
                query = query.eq('atom_type', atom_type)
            
            if min_attention is not None:
                query = query.gte('attention_value', min_attention)
            
            result = query.order('attention_value', desc=True).limit(limit).execute()
            return result.data
            
        except Exception as e:
            logger.error(f"Error finding atoms: {e}")
            return []
    
    def update_attention(self, atom_id: str, attention_delta: float, 
                        reason: Optional[str] = None, allocated_by: Optional[str] = None):
        """
        Update attention value for an atom and log the change.
        
        Args:
            atom_id: UUID of the atom
            attention_delta: Change in attention value
            reason: Reason for attention change
            allocated_by: Process ID that allocated attention
        """
        try:
            # Get current atom
            atom = self.get_atom(atom_id)
            if not atom:
                logger.warning(f"Atom {atom_id} not found for attention update")
                return
            
            # Update attention value
            new_attention = atom['attention_value'] + attention_delta
            self.client.table('atoms').update({
                'attention_value': new_attention
            }).eq('atom_id', atom_id).execute()
            
            # Log attention change
            self.client.table('attention_log').insert({
                'atom_id': atom_id,
                'attention_delta': attention_delta,
                'reason': reason,
                'allocated_by': allocated_by
            }).execute()
            
            logger.debug(f"Updated attention for atom {atom_id}: {attention_delta:+.3f}")
            
        except Exception as e:
            logger.error(f"Error updating attention: {e}")
    
    # Link operations
    
    def create_link(self, source_atom_id: str, target_atom_id: str, 
                   link_type: str, strength: float = 1.0,
                   metadata: Dict[str, Any] = None) -> str:
        """
        Create a link between two atoms.
        
        Args:
            source_atom_id: UUID of source atom
            target_atom_id: UUID of target atom
            link_type: Type of link
            strength: Link strength (0.0 to 1.0)
            metadata: Additional metadata
        
        Returns:
            link_id: UUID of the created link
        """
        try:
            data = {
                'source_atom_id': source_atom_id,
                'target_atom_id': target_atom_id,
                'link_type': link_type,
                'strength': strength,
                'metadata': metadata or {}
            }
            
            result = self.client.table('atom_links').upsert(
                data, 
                on_conflict='source_atom_id,target_atom_id,link_type'
            ).execute()
            
            if result.data:
                link_id = result.data[0]['link_id']
                logger.debug(f"Created link: {link_type} ({source_atom_id} -> {target_atom_id})")
                return link_id
            else:
                raise Exception("Failed to create link")
                
        except Exception as e:
            logger.error(f"Error creating link: {e}")
            raise
    
    def get_atom_links(self, atom_id: str, direction: str = 'both') -> List[Dict[str, Any]]:
        """
        Get all links connected to an atom.
        
        Args:
            atom_id: UUID of the atom
            direction: 'incoming', 'outgoing', or 'both'
        
        Returns:
            List of link dictionaries
        """
        try:
            links = []
            
            if direction in ['outgoing', 'both']:
                result = self.client.table('atom_links').select('*').eq('source_atom_id', atom_id).execute()
                links.extend(result.data)
            
            if direction in ['incoming', 'both']:
                result = self.client.table('atom_links').select('*').eq('target_atom_id', atom_id).execute()
                links.extend(result.data)
            
            return links
            
        except Exception as e:
            logger.error(f"Error getting atom links: {e}")
            return []
    
    # Cognitive process operations
    
    def register_process(self, process_name: str, process_type: str,
                        priority: float = 1.0, metadata: Dict[str, Any] = None) -> str:
        """
        Register a cognitive process.
        
        Args:
            process_name: Name of the process
            process_type: Type of process
            priority: Priority level
            metadata: Additional metadata
        
        Returns:
            process_id: UUID of the registered process
        """
        try:
            data = {
                'process_name': process_name,
                'process_type': process_type,
                'priority': priority,
                'status': 'active',
                'performance_metrics': metadata or {}
            }
            
            result = self.client.table('cognitive_processes').insert(data).execute()
            
            if result.data:
                process_id = result.data[0]['process_id']
                logger.info(f"Registered process: {process_name} (ID: {process_id})")
                return process_id
            else:
                raise Exception("Failed to register process")
                
        except Exception as e:
            logger.error(f"Error registering process: {e}")
            raise
    
    def update_process_status(self, process_id: str, status: str, 
                             is_stuck: bool = False, metrics: Dict[str, Any] = None):
        """
        Update cognitive process status.
        
        Args:
            process_id: UUID of the process
            status: New status
            is_stuck: Whether process is stuck
            metrics: Performance metrics
        """
        try:
            data = {
                'status': status,
                'is_stuck': is_stuck,
                'last_activity': datetime.utcnow().isoformat()
            }
            
            if metrics:
                data['performance_metrics'] = metrics
            
            self.client.table('cognitive_processes').update(data).eq('process_id', process_id).execute()
            logger.debug(f"Updated process {process_id} status: {status}")
            
        except Exception as e:
            logger.error(f"Error updating process status: {e}")
    
    # Pattern operations
    
    def save_pattern(self, pattern_type: str, pattern_data: Dict[str, Any],
                    frequency: int = 1, confidence: float = 0.5,
                    discovered_by: Optional[str] = None) -> str:
        """
        Save a discovered pattern.
        
        Args:
            pattern_type: Type of pattern
            pattern_data: Pattern data as JSON
            frequency: Pattern frequency
            confidence: Pattern confidence
            discovered_by: Process ID that discovered the pattern
        
        Returns:
            pattern_id: UUID of the saved pattern
        """
        try:
            data = {
                'pattern_type': pattern_type,
                'pattern_data': pattern_data,
                'frequency': frequency,
                'confidence': confidence,
                'discovered_by': discovered_by
            }
            
            result = self.client.table('patterns').insert(data).execute()
            
            if result.data:
                pattern_id = result.data[0]['pattern_id']
                logger.debug(f"Saved pattern: {pattern_type} (ID: {pattern_id})")
                return pattern_id
            else:
                raise Exception("Failed to save pattern")
                
        except Exception as e:
            logger.error(f"Error saving pattern: {e}")
            raise
    
    # Synergy event logging
    
    def log_synergy_event(self, event_type: str, source_process_id: Optional[str] = None,
                         target_process_id: Optional[str] = None, 
                         event_data: Dict[str, Any] = None,
                         outcome: Optional[str] = None) -> str:
        """
        Log a cognitive synergy event.
        
        Args:
            event_type: Type of synergy event
            source_process_id: Source process UUID
            target_process_id: Target process UUID
            event_data: Event data as JSON
            outcome: Event outcome
        
        Returns:
            event_id: UUID of the logged event
        """
        try:
            data = {
                'event_type': event_type,
                'source_process_id': source_process_id,
                'target_process_id': target_process_id,
                'event_data': event_data or {},
                'outcome': outcome
            }
            
            result = self.client.table('synergy_events').insert(data).execute()
            
            if result.data:
                event_id = result.data[0]['event_id']
                logger.debug(f"Logged synergy event: {event_type} (ID: {event_id})")
                return event_id
            else:
                raise Exception("Failed to log synergy event")
                
        except Exception as e:
            logger.error(f"Error logging synergy event: {e}")
            raise
    
    # Metrics operations
    
    def record_metric(self, metric_name: str, metric_value: float,
                     metadata: Dict[str, Any] = None):
        """
        Record a synergy metric.
        
        Args:
            metric_name: Name of the metric
            metric_value: Metric value
            metadata: Additional metadata
        """
        try:
            data = {
                'metric_name': metric_name,
                'metric_value': metric_value,
                'metric_metadata': metadata or {}
            }
            
            self.client.table('synergy_metrics').insert(data).execute()
            logger.debug(f"Recorded metric: {metric_name} = {metric_value}")
            
        except Exception as e:
            logger.error(f"Error recording metric: {e}")
    
    def get_metrics_summary(self, metric_name: Optional[str] = None,
                           hours: int = 24) -> List[Dict[str, Any]]:
        """
        Get metrics summary.
        
        Args:
            metric_name: Filter by metric name
            hours: Time window in hours
        
        Returns:
            List of metric summaries
        """
        try:
            # Use the view for efficient querying
            query = self.client.table('synergy_metrics_summary').select('*')
            
            if metric_name:
                query = query.eq('metric_name', metric_name)
            
            result = query.execute()
            return result.data
            
        except Exception as e:
            logger.error(f"Error getting metrics summary: {e}")
            return []


# Example usage
if __name__ == "__main__":
    # Initialize persistence layer
    persistence = HypergraphPersistence()
    
    # Save an atom
    atom_id = persistence.save_atom(
        atom_type='ConceptNode',
        name='cognitive_synergy',
        truth_value=0.95,
        attention_value=0.8,
        metadata={'domain': 'AGI', 'importance': 'high'}
    )
    
    print(f"Created atom with ID: {atom_id}")
    
    # Register a cognitive process
    process_id = persistence.register_process(
        process_name='PatternMiner',
        process_type='mining',
        priority=0.9
    )
    
    print(f"Registered process with ID: {process_id}")
    
    # Record a metric
    persistence.record_metric(
        metric_name='process_efficiency',
        metric_value=0.87,
        metadata={'timestamp': datetime.utcnow().isoformat()}
    )
    
    print("Recorded metric successfully")


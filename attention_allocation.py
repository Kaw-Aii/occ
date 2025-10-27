"""
Attention Allocation Mechanism (ECAN-Inspired)
==============================================

This module implements an Economic Attention Network (ECAN) inspired attention
allocation mechanism for the cognitive architecture.

Features:
- Short-term importance (STI) for immediate relevance
- Long-term importance (LTI) for persistent value
- Attention spreading via hypergraph links
- Forgetting mechanism for low-attention atoms
- Attention-based retrieval optimization
- Hebbian learning for attention patterns

Based on OpenCog's ECAN (Economic Attention Networks) design.

Author: OpenCog Collection Contributors
License: GPL-3.0+
"""

import logging
from typing import Dict, List, Set, Tuple, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
from collections import defaultdict, deque
import math
import random

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class AttentionValue:
    """Attention value for an atom or link."""
    sti: float = 0.0  # Short-term importance
    lti: float = 0.0  # Long-term importance
    vlti: float = 0.0  # Very long-term importance
    
    def total_attention(self) -> float:
        """Calculate total attention value."""
        return self.sti + self.lti + self.vlti
    
    def normalize(self, max_sti: float = 100.0, max_lti: float = 100.0):
        """Normalize attention values to prevent overflow."""
        self.sti = min(self.sti, max_sti)
        self.lti = min(self.lti, max_lti)
        self.vlti = min(self.vlti, max_lti)


@dataclass
class AttentionParameters:
    """Configuration parameters for attention allocation."""
    # STI parameters
    sti_decay_rate: float = 0.1  # Rate of STI decay per cycle
    sti_spread_factor: float = 0.5  # Fraction of STI to spread to neighbors
    min_sti_threshold: float = 0.01  # Minimum STI before forgetting
    
    # LTI parameters
    lti_growth_rate: float = 0.05  # Rate of LTI growth from STI
    lti_decay_rate: float = 0.01  # Rate of LTI decay per cycle
    
    # Spreading parameters
    max_spread_depth: int = 3  # Maximum depth for attention spreading
    spread_threshold: float = 0.1  # Minimum STI to trigger spreading
    
    # Forgetting parameters
    forgetting_threshold: float = 0.001  # Total attention below this triggers forgetting
    forgetting_probability: float = 0.1  # Probability of forgetting low-attention atoms
    
    # Hebbian learning
    hebbian_learning_rate: float = 0.1  # Rate of attention link strengthening
    
    # Economic parameters
    total_sti_budget: float = 1000.0  # Total STI available in the system
    rent_collection_rate: float = 0.05  # Fraction of STI collected as "rent"


@dataclass
class AttentionLink:
    """Link between atoms for attention spreading."""
    source_id: str
    target_id: str
    weight: float = 1.0  # Strength of attention transfer
    usage_count: int = 0  # Number of times this link was used
    last_used: datetime = field(default_factory=datetime.now)


class AttentionBank:
    """
    Central attention allocation system.
    Manages STI/LTI for all atoms and implements attention dynamics.
    """
    
    def __init__(self, params: Optional[AttentionParameters] = None):
        """Initialize attention bank."""
        self.params = params or AttentionParameters()
        
        # Attention values for each atom
        self.attention_values: Dict[str, AttentionValue] = {}
        
        # Attention links between atoms
        self.attention_links: Dict[Tuple[str, str], AttentionLink] = {}
        
        # Attentional focus set (high STI atoms)
        self.attentional_focus: Set[str] = set()
        
        # Forgetting candidates (low attention atoms)
        self.forgetting_candidates: Set[str] = set()
        
        # Statistics
        self.total_sti_allocated: float = 0.0
        self.cycle_count: int = 0
        
        logger.info("AttentionBank initialized")
    
    def get_attention(self, atom_id: str) -> AttentionValue:
        """Get attention value for an atom."""
        if atom_id not in self.attention_values:
            self.attention_values[atom_id] = AttentionValue()
        return self.attention_values[atom_id]
    
    def set_sti(self, atom_id: str, sti: float):
        """Set STI for an atom."""
        av = self.get_attention(atom_id)
        old_sti = av.sti
        av.sti = sti
        
        # Update total STI tracking
        self.total_sti_allocated += (sti - old_sti)
        
        # Update attentional focus
        self._update_attentional_focus(atom_id)
        
        logger.debug(f"Set STI for {atom_id}: {sti:.2f}")
    
    def stimulate(self, atom_id: str, amount: float):
        """Add STI to an atom (stimulation)."""
        av = self.get_attention(atom_id)
        av.sti += amount
        self.total_sti_allocated += amount
        
        # Normalize to prevent overflow
        av.normalize()
        
        self._update_attentional_focus(atom_id)
        
        logger.debug(f"Stimulated {atom_id} by {amount:.2f}, new STI: {av.sti:.2f}")
    
    def add_attention_link(self, source_id: str, target_id: str, weight: float = 1.0):
        """Add or update an attention link between atoms."""
        key = (source_id, target_id)
        
        if key in self.attention_links:
            link = self.attention_links[key]
            link.weight = weight
            link.usage_count += 1
            link.last_used = datetime.now()
        else:
            self.attention_links[key] = AttentionLink(
                source_id=source_id,
                target_id=target_id,
                weight=weight
            )
        
        logger.debug(f"Added attention link: {source_id} -> {target_id} (weight: {weight:.2f})")
    
    def spread_attention(self, atom_id: str, depth: int = 0):
        """
        Spread attention from an atom to its neighbors.
        Implements recursive attention spreading with decay.
        """
        if depth >= self.params.max_spread_depth:
            return
        
        av = self.get_attention(atom_id)
        
        # Only spread if STI is above threshold
        if av.sti < self.params.spread_threshold:
            return
        
        # Find outgoing attention links
        outgoing_links = [
            link for (src, tgt), link in self.attention_links.items()
            if src == atom_id
        ]
        
        if not outgoing_links:
            return
        
        # Calculate total weight for normalization
        total_weight = sum(link.weight for link in outgoing_links)
        
        if total_weight == 0:
            return
        
        # Amount of STI to spread
        spread_amount = av.sti * self.params.sti_spread_factor
        
        # Spread to each neighbor proportionally
        for link in outgoing_links:
            proportion = link.weight / total_weight
            transfer_amount = spread_amount * proportion
            
            # Transfer STI
            target_av = self.get_attention(link.target_id)
            target_av.sti += transfer_amount
            target_av.normalize()
            
            # Update link usage
            link.usage_count += 1
            link.last_used = datetime.now()
            
            # Hebbian learning: strengthen frequently used links
            link.weight += self.params.hebbian_learning_rate
            
            self._update_attentional_focus(link.target_id)
            
            # Recursively spread from target
            self.spread_attention(link.target_id, depth + 1)
        
        # Reduce source STI after spreading
        av.sti *= (1 - self.params.sti_spread_factor)
    
    def decay_attention(self):
        """
        Apply attention decay to all atoms.
        STI decays faster than LTI.
        """
        for atom_id, av in self.attention_values.items():
            # STI decay
            av.sti *= (1 - self.params.sti_decay_rate)
            
            # LTI decay (slower)
            av.lti *= (1 - self.params.lti_decay_rate)
            
            # Transfer some STI to LTI (consolidation)
            sti_to_lti = av.sti * self.params.lti_growth_rate
            av.sti -= sti_to_lti
            av.lti += sti_to_lti
            
            # Update focus and forgetting candidates
            self._update_attentional_focus(atom_id)
            self._update_forgetting_candidates(atom_id)
    
    def collect_rent(self):
        """
        Collect "rent" from all atoms to maintain economic equilibrium.
        Implements attention economy where atoms must maintain relevance.
        """
        total_collected = 0.0
        
        for atom_id, av in self.attention_values.items():
            rent = av.sti * self.params.rent_collection_rate
            av.sti -= rent
            total_collected += rent
        
        self.total_sti_allocated -= total_collected
        
        logger.debug(f"Collected {total_collected:.2f} STI as rent")
        
        return total_collected
    
    def redistribute_attention(self, amount: float, target_atoms: List[str]):
        """
        Redistribute collected attention to target atoms.
        Used to inject attention into important atoms.
        """
        if not target_atoms:
            return
        
        per_atom = amount / len(target_atoms)
        
        for atom_id in target_atoms:
            self.stimulate(atom_id, per_atom)
    
    def get_attentional_focus(self, top_k: int = 10) -> List[Tuple[str, float]]:
        """
        Get atoms in attentional focus (highest STI).
        """
        sorted_atoms = sorted(
            self.attention_values.items(),
            key=lambda x: x[1].sti,
            reverse=True
        )
        
        return [(atom_id, av.sti) for atom_id, av in sorted_atoms[:top_k]]
    
    def get_forgetting_candidates(self) -> List[str]:
        """Get atoms that are candidates for forgetting."""
        return list(self.forgetting_candidates)
    
    def forget_atom(self, atom_id: str):
        """Remove an atom from attention tracking (forgetting)."""
        if atom_id in self.attention_values:
            av = self.attention_values[atom_id]
            self.total_sti_allocated -= av.sti
            del self.attention_values[atom_id]
        
        if atom_id in self.attentional_focus:
            self.attentional_focus.remove(atom_id)
        
        if atom_id in self.forgetting_candidates:
            self.forgetting_candidates.remove(atom_id)
        
        # Remove attention links involving this atom
        links_to_remove = [
            key for key in self.attention_links.keys()
            if key[0] == atom_id or key[1] == atom_id
        ]
        
        for key in links_to_remove:
            del self.attention_links[key]
        
        logger.debug(f"Forgot atom: {atom_id}")
    
    def run_attention_cycle(self, stimulated_atoms: Optional[List[str]] = None):
        """
        Run one cycle of attention dynamics.
        
        Steps:
        1. Stimulate relevant atoms
        2. Spread attention
        3. Decay attention
        4. Collect rent
        5. Redistribute attention
        6. Forget low-attention atoms
        """
        self.cycle_count += 1
        logger.info(f"Running attention cycle {self.cycle_count}")
        
        # 1. Stimulate atoms if provided
        if stimulated_atoms:
            for atom_id in stimulated_atoms:
                self.stimulate(atom_id, 10.0)  # Base stimulation amount
        
        # 2. Spread attention from high-STI atoms
        focus_atoms = [atom_id for atom_id, _ in self.get_attentional_focus(top_k=20)]
        for atom_id in focus_atoms:
            self.spread_attention(atom_id)
        
        # 3. Decay all attention
        self.decay_attention()
        
        # 4. Collect rent
        collected = self.collect_rent()
        
        # 5. Redistribute to important atoms
        if collected > 0 and focus_atoms:
            self.redistribute_attention(collected, focus_atoms[:5])
        
        # 6. Forget low-attention atoms
        candidates = self.get_forgetting_candidates()
        for atom_id in candidates:
            if random.random() < self.params.forgetting_probability:
                self.forget_atom(atom_id)
        
        logger.info(f"Cycle {self.cycle_count} complete. Total STI: {self.total_sti_allocated:.2f}")
    
    def _update_attentional_focus(self, atom_id: str):
        """Update attentional focus set based on STI."""
        av = self.get_attention(atom_id)
        
        # Add to focus if STI is high
        if av.sti > 10.0:  # Threshold for attentional focus
            self.attentional_focus.add(atom_id)
        else:
            self.attentional_focus.discard(atom_id)
    
    def _update_forgetting_candidates(self, atom_id: str):
        """Update forgetting candidates based on total attention."""
        av = self.get_attention(atom_id)
        
        if av.total_attention() < self.params.forgetting_threshold:
            self.forgetting_candidates.add(atom_id)
        else:
            self.forgetting_candidates.discard(atom_id)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get attention bank statistics."""
        return {
            "cycle_count": self.cycle_count,
            "total_atoms": len(self.attention_values),
            "total_sti_allocated": self.total_sti_allocated,
            "attentional_focus_size": len(self.attentional_focus),
            "forgetting_candidates": len(self.forgetting_candidates),
            "attention_links": len(self.attention_links),
            "avg_sti": self.total_sti_allocated / max(len(self.attention_values), 1)
        }


class AttentionBasedRetrieval:
    """
    Attention-based retrieval system for efficient knowledge access.
    """
    
    def __init__(self, attention_bank: AttentionBank):
        """Initialize attention-based retrieval."""
        self.attention_bank = attention_bank
        logger.info("AttentionBasedRetrieval initialized")
    
    def retrieve_by_attention(self, min_sti: float = 1.0, limit: int = 100) -> List[str]:
        """Retrieve atoms with STI above threshold."""
        results = [
            atom_id for atom_id, av in self.attention_bank.attention_values.items()
            if av.sti >= min_sti
        ]
        
        # Sort by STI descending
        results.sort(key=lambda x: self.attention_bank.get_attention(x).sti, reverse=True)
        
        return results[:limit]
    
    def retrieve_by_lti(self, min_lti: float = 1.0, limit: int = 100) -> List[str]:
        """Retrieve atoms with high long-term importance."""
        results = [
            atom_id for atom_id, av in self.attention_bank.attention_values.items()
            if av.lti >= min_lti
        ]
        
        results.sort(key=lambda x: self.attention_bank.get_attention(x).lti, reverse=True)
        
        return results[:limit]
    
    def retrieve_neighbors(self, atom_id: str, max_depth: int = 2) -> Set[str]:
        """Retrieve atoms connected to given atom via attention links."""
        visited = set()
        queue = deque([(atom_id, 0)])
        
        while queue:
            current_id, depth = queue.popleft()
            
            if current_id in visited or depth > max_depth:
                continue
            
            visited.add(current_id)
            
            # Find neighbors
            neighbors = [
                target for (source, target) in self.attention_bank.attention_links.keys()
                if source == current_id
            ]
            
            for neighbor in neighbors:
                if neighbor not in visited:
                    queue.append((neighbor, depth + 1))
        
        return visited


# Example usage
if __name__ == "__main__":
    # Initialize attention bank
    params = AttentionParameters(
        sti_decay_rate=0.1,
        sti_spread_factor=0.5,
        lti_growth_rate=0.05
    )
    
    bank = AttentionBank(params)
    
    # Create some atoms with attention
    atoms = ["concept_1", "concept_2", "concept_3", "concept_4", "concept_5"]
    
    for atom in atoms:
        bank.stimulate(atom, 20.0)
    
    # Add attention links
    bank.add_attention_link("concept_1", "concept_2", weight=0.8)
    bank.add_attention_link("concept_1", "concept_3", weight=0.6)
    bank.add_attention_link("concept_2", "concept_4", weight=0.7)
    bank.add_attention_link("concept_3", "concept_5", weight=0.5)
    
    # Run attention cycles
    for i in range(5):
        bank.run_attention_cycle(stimulated_atoms=["concept_1"])
        
        # Print statistics
        stats = bank.get_statistics()
        print(f"\nCycle {i+1} Statistics:")
        print(f"  Total atoms: {stats['total_atoms']}")
        print(f"  Total STI: {stats['total_sti_allocated']:.2f}")
        print(f"  Attentional focus: {stats['attentional_focus_size']}")
        
        # Print top attention atoms
        focus = bank.get_attentional_focus(top_k=3)
        print(f"  Top attention atoms:")
        for atom_id, sti in focus:
            print(f"    {atom_id}: {sti:.2f}")
    
    # Attention-based retrieval
    retrieval = AttentionBasedRetrieval(bank)
    high_attention = retrieval.retrieve_by_attention(min_sti=5.0)
    print(f"\nHigh attention atoms: {high_attention}")


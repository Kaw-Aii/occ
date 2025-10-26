"""
Hypergraph Knowledge Bridge - Universal Knowledge Translation Layer
===================================================================

This module implements a universal knowledge translation layer that bridges
different knowledge representations across cognitive paradigms.

The bridge enables:
- Symbolic ↔ Neural knowledge translation
- Pattern-based knowledge retrieval
- Temporal knowledge tracking
- Multi-modal knowledge integration
- Cross-paradigm knowledge sharing

This serves as the common substrate for cognitive synergy, allowing different
AI paradigms to share knowledge through the hypergraph memory structure.

Author: OpenCog Collection Contributors
License: GPL-3.0+
"""

import logging
from typing import Dict, List, Any, Optional, Set, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime
from collections import defaultdict, deque
import numpy as np
import hashlib
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class Atom:
    """
    Basic unit of knowledge in the hypergraph.
    Represents a concept, entity, or value.
    """
    atom_id: str
    atom_type: str  # 'concept', 'predicate', 'value', 'variable'
    name: str
    truth_value: Tuple[float, float] = (1.0, 1.0)  # (strength, confidence)
    attention_value: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    creation_time: datetime = field(default_factory=datetime.now)


@dataclass
class Link:
    """
    Relationship between atoms in the hypergraph.
    Represents n-ary relationships.
    """
    link_id: str
    link_type: str  # 'inheritance', 'similarity', 'evaluation', 'implication'
    outgoing: List[str]  # List of atom_ids
    truth_value: Tuple[float, float] = (1.0, 1.0)
    attention_value: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    creation_time: datetime = field(default_factory=datetime.now)


@dataclass
class Pattern:
    """
    A pattern discovered in the hypergraph.
    Can be used for retrieval and knowledge transfer.
    """
    pattern_id: str
    pattern_type: str  # 'structural', 'temporal', 'semantic'
    structure: Dict[str, Any]
    frequency: int = 1
    confidence: float = 1.0
    source_paradigm: str = "unknown"
    instances: List[str] = field(default_factory=list)


class HypergraphMemory:
    """
    Core hypergraph memory structure.
    Stores atoms and links with efficient retrieval.
    """
    
    def __init__(self):
        self.atoms: Dict[str, Atom] = {}
        self.links: Dict[str, Link] = {}
        self.incoming_sets: Dict[str, Set[str]] = defaultdict(set)  # atom_id -> link_ids
        self.type_indices: Dict[str, Set[str]] = defaultdict(set)  # type -> atom/link_ids
        
        logger.info("HypergraphMemory initialized")
    
    def add_atom(self, atom: Atom) -> str:
        """Add an atom to the hypergraph."""
        self.atoms[atom.atom_id] = atom
        self.type_indices[atom.atom_type].add(atom.atom_id)
        logger.debug(f"Atom added: {atom.name} ({atom.atom_type})")
        return atom.atom_id
    
    def add_link(self, link: Link) -> str:
        """Add a link to the hypergraph."""
        self.links[link.link_id] = link
        self.type_indices[link.link_type].add(link.link_id)
        
        # Update incoming sets
        for atom_id in link.outgoing:
            self.incoming_sets[atom_id].add(link.link_id)
        
        logger.debug(f"Link added: {link.link_type} with {len(link.outgoing)} atoms")
        return link.link_id
    
    def get_atom(self, atom_id: str) -> Optional[Atom]:
        """Retrieve an atom by ID."""
        return self.atoms.get(atom_id)
    
    def get_link(self, link_id: str) -> Optional[Link]:
        """Retrieve a link by ID."""
        return self.links.get(link_id)
    
    def get_incoming_links(self, atom_id: str) -> List[Link]:
        """Get all links pointing to an atom."""
        link_ids = self.incoming_sets.get(atom_id, set())
        return [self.links[lid] for lid in link_ids if lid in self.links]
    
    def find_atoms_by_type(self, atom_type: str) -> List[Atom]:
        """Find all atoms of a given type."""
        atom_ids = self.type_indices.get(atom_type, set())
        return [self.atoms[aid] for aid in atom_ids if aid in self.atoms]
    
    def find_links_by_type(self, link_type: str) -> List[Link]:
        """Find all links of a given type."""
        link_ids = self.type_indices.get(link_type, set())
        return [self.links[lid] for lid in link_ids if lid in self.links]


class SymbolicNeuralTranslator:
    """
    Translates between symbolic and neural representations.
    
    Symbolic: Discrete atoms and links with logical relationships
    Neural: Continuous vector embeddings and similarity metrics
    """
    
    def __init__(self, embedding_dim: int = 128):
        self.embedding_dim = embedding_dim
        self.atom_embeddings: Dict[str, np.ndarray] = {}
        self.concept_to_vector: Dict[str, np.ndarray] = {}
        self.vector_to_concept: Dict[str, str] = {}
        
        logger.info(f"SymbolicNeuralTranslator initialized (dim={embedding_dim})")
    
    def symbolic_to_neural(self, atom: Atom) -> np.ndarray:
        """Convert a symbolic atom to a neural embedding."""
        if atom.atom_id in self.atom_embeddings:
            return self.atom_embeddings[atom.atom_id]
        
        # Generate embedding from atom properties
        # Simple hash-based embedding (in practice, use learned embeddings)
        hash_val = int(hashlib.md5(atom.name.encode()).hexdigest(), 16)
        np.random.seed(hash_val % (2**32))
        embedding = np.random.randn(self.embedding_dim)
        
        # Normalize
        embedding = embedding / (np.linalg.norm(embedding) + 1e-8)
        
        # Weight by truth value
        strength, confidence = atom.truth_value
        embedding = embedding * strength * confidence
        
        self.atom_embeddings[atom.atom_id] = embedding
        return embedding
    
    def neural_to_symbolic(self, embedding: np.ndarray,
                          threshold: float = 0.7) -> List[Atom]:
        """Find symbolic atoms similar to a neural embedding."""
        similar_atoms = []
        
        for atom_id, atom_emb in self.atom_embeddings.items():
            similarity = np.dot(embedding, atom_emb) / (
                np.linalg.norm(embedding) * np.linalg.norm(atom_emb) + 1e-8
            )
            
            if similarity >= threshold:
                # Retrieve atom (would need hypergraph reference)
                similar_atoms.append((atom_id, similarity))
        
        return similar_atoms
    
    def link_to_relation_matrix(self, link: Link,
                               hypergraph: HypergraphMemory) -> np.ndarray:
        """Convert a symbolic link to a neural relation matrix."""
        # Get embeddings for all atoms in the link
        embeddings = []
        for atom_id in link.outgoing:
            atom = hypergraph.get_atom(atom_id)
            if atom:
                emb = self.symbolic_to_neural(atom)
                embeddings.append(emb)
        
        if not embeddings:
            return np.zeros((self.embedding_dim, self.embedding_dim))
        
        # Create relation matrix (outer product of embeddings)
        relation = np.zeros((self.embedding_dim, self.embedding_dim))
        for i, emb_i in enumerate(embeddings):
            for j, emb_j in enumerate(embeddings):
                relation += np.outer(emb_i, emb_j)
        
        # Normalize
        relation = relation / (len(embeddings) ** 2)
        
        return relation


class PatternMiner:
    """
    Discovers patterns in the hypergraph for cross-paradigm knowledge sharing.
    """
    
    def __init__(self):
        self.patterns: Dict[str, Pattern] = {}
        self.pattern_index: Dict[str, Set[str]] = defaultdict(set)
        
        logger.info("PatternMiner initialized")
    
    def mine_structural_patterns(self, hypergraph: HypergraphMemory,
                                 min_frequency: int = 2) -> List[Pattern]:
        """Mine structural patterns from the hypergraph."""
        discovered_patterns = []
        
        # Find common link structures
        link_type_structures: Dict[str, List[List[str]]] = defaultdict(list)
        
        for link in hypergraph.links.values():
            # Create structure signature
            structure_sig = (
                link.link_type,
                tuple(hypergraph.get_atom(aid).atom_type 
                     for aid in link.outgoing if hypergraph.get_atom(aid))
            )
            link_type_structures[str(structure_sig)].append(link.outgoing)
        
        # Identify frequent patterns
        for structure_sig, instances in link_type_structures.items():
            if len(instances) >= min_frequency:
                pattern = Pattern(
                    pattern_id=f"pat_{len(self.patterns)}",
                    pattern_type="structural",
                    structure={'signature': structure_sig, 'arity': len(instances[0])},
                    frequency=len(instances),
                    confidence=len(instances) / len(hypergraph.links),
                    instances=[str(inst) for inst in instances]
                )
                self.patterns[pattern.pattern_id] = pattern
                discovered_patterns.append(pattern)
        
        logger.info(f"Discovered {len(discovered_patterns)} structural patterns")
        return discovered_patterns
    
    def mine_temporal_patterns(self, hypergraph: HypergraphMemory,
                              time_window: float = 60.0) -> List[Pattern]:
        """Mine temporal patterns (sequences of knowledge creation)."""
        discovered_patterns = []
        
        # Sort atoms by creation time
        sorted_atoms = sorted(hypergraph.atoms.values(),
                            key=lambda a: a.creation_time)
        
        # Find temporal sequences
        sequences: Dict[str, List[str]] = defaultdict(list)
        
        for i in range(len(sorted_atoms) - 1):
            atom_a = sorted_atoms[i]
            atom_b = sorted_atoms[i + 1]
            
            time_diff = (atom_b.creation_time - atom_a.creation_time).total_seconds()
            
            if time_diff <= time_window:
                seq_key = f"{atom_a.atom_type}->{atom_b.atom_type}"
                sequences[seq_key].append((atom_a.atom_id, atom_b.atom_id))
        
        # Create patterns from frequent sequences
        for seq_key, instances in sequences.items():
            if len(instances) >= 2:
                pattern = Pattern(
                    pattern_id=f"temp_{len(self.patterns)}",
                    pattern_type="temporal",
                    structure={'sequence': seq_key, 'window': time_window},
                    frequency=len(instances),
                    confidence=1.0,
                    instances=[f"{a}->{b}" for a, b in instances]
                )
                self.patterns[pattern.pattern_id] = pattern
                discovered_patterns.append(pattern)
        
        logger.info(f"Discovered {len(discovered_patterns)} temporal patterns")
        return discovered_patterns


class HypergraphKnowledgeBridge:
    """
    Main bridge coordinating all knowledge translation and integration.
    
    Provides unified interface for:
    - Adding knowledge from different paradigms
    - Translating between representations
    - Mining cross-paradigm patterns
    - Retrieving relevant knowledge
    """
    
    def __init__(self, embedding_dim: int = 128):
        self.hypergraph = HypergraphMemory()
        self.translator = SymbolicNeuralTranslator(embedding_dim)
        self.pattern_miner = PatternMiner()
        
        # Track knowledge sources
        self.paradigm_contributions: Dict[str, int] = defaultdict(int)
        
        logger.info("HypergraphKnowledgeBridge initialized - universal knowledge layer ready")
    
    def add_symbolic_knowledge(self, concept: str, concept_type: str,
                              truth_value: Tuple[float, float] = (1.0, 1.0),
                              paradigm: str = "symbolic") -> str:
        """Add symbolic knowledge to the hypergraph."""
        atom_id = f"atom_{len(self.hypergraph.atoms)}"
        atom = Atom(
            atom_id=atom_id,
            atom_type=concept_type,
            name=concept,
            truth_value=truth_value,
            metadata={'paradigm': paradigm}
        )
        
        self.hypergraph.add_atom(atom)
        self.paradigm_contributions[paradigm] += 1
        
        return atom_id
    
    def add_neural_knowledge(self, embedding: np.ndarray,
                            concept_name: str,
                            paradigm: str = "neural") -> str:
        """Add neural knowledge (embedding) to the hypergraph."""
        # Convert embedding to symbolic representation
        atom_id = f"atom_{len(self.hypergraph.atoms)}"
        
        # Compute confidence from embedding norm
        confidence = min(np.linalg.norm(embedding), 1.0)
        
        atom = Atom(
            atom_id=atom_id,
            atom_type="neural_concept",
            name=concept_name,
            truth_value=(1.0, confidence),
            metadata={'paradigm': paradigm, 'has_embedding': True}
        )
        
        self.hypergraph.add_atom(atom)
        self.translator.atom_embeddings[atom_id] = embedding
        self.paradigm_contributions[paradigm] += 1
        
        return atom_id
    
    def add_relationship(self, relation_type: str, atoms: List[str],
                        truth_value: Tuple[float, float] = (1.0, 1.0)) -> str:
        """Add a relationship between atoms."""
        link_id = f"link_{len(self.hypergraph.links)}"
        link = Link(
            link_id=link_id,
            link_type=relation_type,
            outgoing=atoms,
            truth_value=truth_value
        )
        
        self.hypergraph.add_link(link)
        return link_id
    
    def translate_to_neural(self, atom_id: str) -> Optional[np.ndarray]:
        """Translate a symbolic atom to neural representation."""
        atom = self.hypergraph.get_atom(atom_id)
        if not atom:
            return None
        
        return self.translator.symbolic_to_neural(atom)
    
    def find_similar_concepts(self, embedding: np.ndarray,
                             threshold: float = 0.7) -> List[Tuple[str, float]]:
        """Find symbolic concepts similar to a neural embedding."""
        return self.translator.neural_to_symbolic(embedding, threshold)
    
    def mine_patterns(self, pattern_type: str = "structural") -> List[Pattern]:
        """Mine patterns from the hypergraph."""
        if pattern_type == "structural":
            return self.pattern_miner.mine_structural_patterns(self.hypergraph)
        elif pattern_type == "temporal":
            return self.pattern_miner.mine_temporal_patterns(self.hypergraph)
        else:
            return []
    
    def get_knowledge_statistics(self) -> Dict[str, Any]:
        """Get statistics about knowledge in the bridge."""
        return {
            'total_atoms': len(self.hypergraph.atoms),
            'total_links': len(self.hypergraph.links),
            'total_patterns': len(self.pattern_miner.patterns),
            'paradigm_contributions': dict(self.paradigm_contributions),
            'atom_types': {
                atom_type: len(atom_ids)
                for atom_type, atom_ids in self.hypergraph.type_indices.items()
                if atom_ids and atom_type in [a.atom_type for a in self.hypergraph.atoms.values()]
            }
        }


# Example usage and demonstration
if __name__ == "__main__":
    print("Hypergraph Knowledge Bridge - Universal Knowledge Translation")
    print("=" * 70)
    
    # Initialize bridge
    bridge = HypergraphKnowledgeBridge(embedding_dim=64)
    
    # Add symbolic knowledge
    print("\nAdding symbolic knowledge...")
    cat_id = bridge.add_symbolic_knowledge("cat", "concept", paradigm="symbolic")
    animal_id = bridge.add_symbolic_knowledge("animal", "concept", paradigm="symbolic")
    mammal_id = bridge.add_symbolic_knowledge("mammal", "concept", paradigm="symbolic")
    
    # Add relationships
    bridge.add_relationship("inheritance", [cat_id, mammal_id])
    bridge.add_relationship("inheritance", [mammal_id, animal_id])
    
    # Add neural knowledge
    print("Adding neural knowledge...")
    dog_embedding = np.random.randn(64)
    dog_embedding = dog_embedding / np.linalg.norm(dog_embedding)
    dog_id = bridge.add_neural_knowledge(dog_embedding, "dog", paradigm="neural")
    
    # Translate symbolic to neural
    print("\nTranslating symbolic to neural...")
    cat_embedding = bridge.translate_to_neural(cat_id)
    print(f"Cat embedding shape: {cat_embedding.shape}")
    print(f"Cat embedding norm: {np.linalg.norm(cat_embedding):.3f}")
    
    # Find similar concepts
    print("\nFinding concepts similar to dog embedding...")
    similar = bridge.find_similar_concepts(dog_embedding, threshold=0.5)
    print(f"Found {len(similar)} similar concepts")
    
    # Mine patterns
    print("\nMining structural patterns...")
    patterns = bridge.mine_patterns("structural")
    print(f"Discovered {len(patterns)} patterns")
    
    # Get statistics
    print("\n" + "=" * 70)
    print("Knowledge Statistics:")
    stats = bridge.get_knowledge_statistics()
    print(json.dumps(stats, indent=2))


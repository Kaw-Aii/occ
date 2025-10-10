#!/usr/bin/env python3
"""
Neural-Symbolic Integration Module for OpenCog Collection
========================================================

This module implements advanced neural-symbolic integration capabilities,
enabling seamless collaboration between neural networks and symbolic reasoning
systems through shared representations and bidirectional translation.

Key Features:
- Neural network to symbolic rule extraction
- Symbolic knowledge injection into neural networks
- Hybrid reasoning combining both paradigms
- Attention-guided neural-symbolic fusion
- Differentiable symbolic reasoning
- Neuro-symbolic pattern learning

Author: OpenCog Collection Contributors
License: GPL-3.0+
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Set, Union
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import json
import logging
from datetime import datetime
import threading
from collections import defaultdict, deque
import re

# Neural network components
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import Dataset, DataLoader
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("PyTorch not available. Using numpy-based neural network implementation.")

from cognitive_synergy_framework import (
    CognitiveSynergyEngine, CognitiveProcess, Atom, 
    HypergraphMemory, PatternMiner
)

logger = logging.getLogger(__name__)


@dataclass
class SymbolicRule:
    """Represents a symbolic rule extracted from neural networks or reasoning."""
    premise: List[str]
    conclusion: str
    confidence: float
    support: int = 0
    neural_activation: Optional[float] = None
    extraction_method: str = "unknown"
    
    def to_logic_string(self) -> str:
        """Convert rule to logical string representation."""
        premise_str = " ∧ ".join(self.premise)
        return f"({premise_str}) → {self.conclusion} [{self.confidence:.3f}]"


@dataclass
class NeuralPattern:
    """Represents a pattern learned by neural networks."""
    input_pattern: np.ndarray
    output_pattern: np.ndarray
    activation_pattern: np.ndarray
    symbolic_interpretation: Optional[str] = None
    confidence: float = 0.0


class SymbolicKnowledgeBase:
    """
    Knowledge base for storing and managing symbolic rules and facts.
    """
    
    def __init__(self):
        self.rules: List[SymbolicRule] = []
        self.facts: Set[str] = set()
        self.predicates: Dict[str, List[str]] = defaultdict(list)
        self.rule_index: Dict[str, List[int]] = defaultdict(list)
        
    def add_rule(self, rule: SymbolicRule):
        """Add a symbolic rule to the knowledge base."""
        rule_id = len(self.rules)
        self.rules.append(rule)
        
        # Index rule by conclusion
        self.rule_index[rule.conclusion].append(rule_id)
        
        # Index rule by premises
        for premise in rule.premise:
            self.rule_index[premise].append(rule_id)
    
    def add_fact(self, fact: str):
        """Add a fact to the knowledge base."""
        self.facts.add(fact)
        
        # Extract predicate and arguments
        if "(" in fact and ")" in fact:
            predicate = fact.split("(")[0]
            args = fact.split("(")[1].split(")")[0].split(",")
            self.predicates[predicate].extend([arg.strip() for arg in args])
    
    def query(self, query: str) -> List[Tuple[str, float]]:
        """Query the knowledge base for facts and derived conclusions."""
        results = []
        
        # Direct fact lookup
        if query in self.facts:
            results.append((query, 1.0))
        
        # Rule-based inference
        applicable_rules = self.rule_index.get(query, [])
        for rule_id in applicable_rules:
            rule = self.rules[rule_id]
            
            # Check if all premises are satisfied
            premise_satisfaction = []
            for premise in rule.premise:
                if premise in self.facts:
                    premise_satisfaction.append(1.0)
                else:
                    # Recursive query for premises
                    sub_results = self.query(premise)
                    if sub_results:
                        premise_satisfaction.append(max(conf for _, conf in sub_results))
                    else:
                        premise_satisfaction.append(0.0)
            
            # Calculate rule confidence
            if premise_satisfaction and min(premise_satisfaction) > 0.5:
                rule_confidence = rule.confidence * min(premise_satisfaction)
                results.append((rule.conclusion, rule_confidence))
        
        return results
    
    def get_rules_by_predicate(self, predicate: str) -> List[SymbolicRule]:
        """Get all rules involving a specific predicate."""
        relevant_rules = []
        for rule in self.rules:
            if (predicate in rule.conclusion or 
                any(predicate in premise for premise in rule.premise)):
                relevant_rules.append(rule)
        return relevant_rules


class NeuralSymbolicNetwork:
    """
    Neural network with symbolic reasoning capabilities.
    """
    
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        if TORCH_AVAILABLE:
            self.network = self._create_torch_network()
        else:
            self.network = self._create_numpy_network()
        
        self.symbolic_kb = SymbolicKnowledgeBase()
        self.activation_history = deque(maxlen=1000)
        self.rule_extraction_threshold = 0.8
        
    def _create_torch_network(self):
        """Create PyTorch-based neural network."""
        class SymbolicNet(nn.Module):
            def __init__(self, input_size, hidden_size, output_size):
                super().__init__()
                self.fc1 = nn.Linear(input_size, hidden_size)
                self.fc2 = nn.Linear(hidden_size, hidden_size)
                self.fc3 = nn.Linear(hidden_size, output_size)
                self.dropout = nn.Dropout(0.2)
                
            def forward(self, x):
                h1 = F.relu(self.fc1(x))
                h1 = self.dropout(h1)
                h2 = F.relu(self.fc2(h1))
                h2 = self.dropout(h2)
                output = torch.sigmoid(self.fc3(h2))
                return output, h1, h2
        
        return SymbolicNet(self.input_size, self.hidden_size, self.output_size)
    
    def _create_numpy_network(self):
        """Create numpy-based neural network."""
        class NumpyNet:
            def __init__(self, input_size, hidden_size, output_size):
                # Initialize weights
                self.W1 = np.random.randn(input_size, hidden_size) * 0.1
                self.b1 = np.zeros((1, hidden_size))
                self.W2 = np.random.randn(hidden_size, hidden_size) * 0.1
                self.b2 = np.zeros((1, hidden_size))
                self.W3 = np.random.randn(hidden_size, output_size) * 0.1
                self.b3 = np.zeros((1, output_size))
                
            def forward(self, x):
                # Forward pass
                z1 = np.dot(x, self.W1) + self.b1
                h1 = np.maximum(0, z1)  # ReLU
                
                z2 = np.dot(h1, self.W2) + self.b2
                h2 = np.maximum(0, z2)  # ReLU
                
                z3 = np.dot(h2, self.W3) + self.b3
                output = 1 / (1 + np.exp(-z3))  # Sigmoid
                
                return output, h1, h2
            
            def backward(self, x, y, output, h1, h2, learning_rate=0.01):
                # Simple backpropagation
                m = x.shape[0]
                
                # Output layer gradients
                dz3 = output - y
                dW3 = np.dot(h2.T, dz3) / m
                db3 = np.sum(dz3, axis=0, keepdims=True) / m
                
                # Hidden layer 2 gradients
                dh2 = np.dot(dz3, self.W3.T)
                dz2 = dh2 * (h2 > 0)  # ReLU derivative
                dW2 = np.dot(h1.T, dz2) / m
                db2 = np.sum(dz2, axis=0, keepdims=True) / m
                
                # Hidden layer 1 gradients
                dh1 = np.dot(dz2, self.W2.T)
                dz1 = dh1 * (h1 > 0)  # ReLU derivative
                dW1 = np.dot(x.T, dz1) / m
                db1 = np.sum(dz1, axis=0, keepdims=True) / m
                
                # Update weights
                self.W3 -= learning_rate * dW3
                self.b3 -= learning_rate * db3
                self.W2 -= learning_rate * dW2
                self.b2 -= learning_rate * db2
                self.W1 -= learning_rate * dW1
                self.b1 -= learning_rate * db1
        
        return NumpyNet(self.input_size, self.hidden_size, self.output_size)
    
    def forward(self, x: np.ndarray) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """Forward pass through the network."""
        if TORCH_AVAILABLE and isinstance(self.network, nn.Module):
            with torch.no_grad():
                x_tensor = torch.FloatTensor(x)
                output, h1, h2 = self.network(x_tensor)
                activations = {
                    'hidden1': h1.numpy(),
                    'hidden2': h2.numpy(),
                    'output': output.numpy()
                }
                return output.numpy(), activations
        else:
            output, h1, h2 = self.network.forward(x)
            activations = {
                'hidden1': h1,
                'hidden2': h2,
                'output': output
            }
            return output, activations
    
    def train_step(self, x: np.ndarray, y: np.ndarray) -> float:
        """Perform one training step."""
        if TORCH_AVAILABLE and isinstance(self.network, nn.Module):
            # PyTorch training
            self.network.train()
            x_tensor = torch.FloatTensor(x)
            y_tensor = torch.FloatTensor(y)
            
            optimizer = torch.optim.Adam(self.network.parameters(), lr=0.001)
            criterion = nn.MSELoss()
            
            optimizer.zero_grad()
            output, _, _ = self.network(x_tensor)
            loss = criterion(output, y_tensor)
            loss.backward()
            optimizer.step()
            
            return loss.item()
        else:
            # Numpy training
            output, h1, h2 = self.network.forward(x)
            self.network.backward(x, y, output, h1, h2)
            loss = np.mean((output - y) ** 2)
            return loss
    
    def extract_symbolic_rules(self, input_data: np.ndarray, 
                             output_data: np.ndarray,
                             feature_names: List[str],
                             output_names: List[str]) -> List[SymbolicRule]:
        """Extract symbolic rules from trained neural network."""
        rules = []
        
        # Analyze network activations for different inputs
        for i, (x, y) in enumerate(zip(input_data, output_data)):
            output, activations = self.forward(x.reshape(1, -1))
            
            # Find highly activated neurons
            h1_active = np.where(activations['hidden1'][0] > self.rule_extraction_threshold)[0]
            h2_active = np.where(activations['hidden2'][0] > self.rule_extraction_threshold)[0]
            
            # Create premises based on input features
            premises = []
            for j, feature_val in enumerate(x):
                if feature_val > 0.5:  # Threshold for binary features
                    premises.append(f"{feature_names[j]}(high)")
                elif feature_val < -0.5:
                    premises.append(f"{feature_names[j]}(low)")
            
            # Create conclusions based on output
            conclusions = []
            for j, output_val in enumerate(output[0]):
                if output_val > 0.7:  # High confidence threshold
                    conclusions.append(f"{output_names[j]}(true)")
            
            # Create rules
            if premises and conclusions:
                for conclusion in conclusions:
                    rule = SymbolicRule(
                        premise=premises,
                        conclusion=conclusion,
                        confidence=float(np.max(output[0])),
                        neural_activation=float(np.mean(activations['hidden2'][0])),
                        extraction_method="activation_analysis"
                    )
                    rules.append(rule)
        
        return rules
    
    def inject_symbolic_knowledge(self, rules: List[SymbolicRule]):
        """Inject symbolic knowledge into the neural network."""
        # Add rules to symbolic knowledge base
        for rule in rules:
            self.symbolic_kb.add_rule(rule)
        
        # Create training data from symbolic rules
        training_inputs = []
        training_outputs = []
        
        for rule in rules:
            # Convert symbolic premises to neural input
            input_vector = self._symbolic_to_neural_input(rule.premise)
            output_vector = self._symbolic_to_neural_output(rule.conclusion)
            
            if input_vector is not None and output_vector is not None:
                training_inputs.append(input_vector)
                training_outputs.append(output_vector)
        
        if training_inputs:
            # Fine-tune network with symbolic knowledge
            X = np.array(training_inputs)
            y = np.array(training_outputs)
            
            for epoch in range(10):  # Limited fine-tuning
                loss = self.train_step(X, y)
                if epoch % 5 == 0:
                    logger.info(f"Symbolic injection epoch {epoch}, loss: {loss:.4f}")
    
    def _symbolic_to_neural_input(self, premises: List[str]) -> Optional[np.ndarray]:
        """Convert symbolic premises to neural network input."""
        # This is a simplified conversion - in practice, this would be more sophisticated
        input_vector = np.zeros(self.input_size)
        
        for i, premise in enumerate(premises):
            if i < self.input_size:
                # Simple hash-based encoding
                hash_val = hash(premise) % 1000 / 1000.0
                input_vector[i] = hash_val
        
        return input_vector
    
    def _symbolic_to_neural_output(self, conclusion: str) -> Optional[np.ndarray]:
        """Convert symbolic conclusion to neural network output."""
        output_vector = np.zeros(self.output_size)
        
        # Simple encoding based on conclusion content
        if "true" in conclusion.lower():
            output_vector[0] = 1.0
        elif "false" in conclusion.lower():
            output_vector[0] = 0.0
        else:
            # Hash-based encoding for other conclusions
            hash_val = hash(conclusion) % 1000 / 1000.0
            output_vector[0] = hash_val
        
        return output_vector
    
    def hybrid_reasoning(self, query: str, input_data: np.ndarray) -> Tuple[List[Tuple[str, float]], np.ndarray]:
        """Perform hybrid neural-symbolic reasoning."""
        # Symbolic reasoning
        symbolic_results = self.symbolic_kb.query(query)
        
        # Neural reasoning
        neural_output, activations = self.forward(input_data.reshape(1, -1))
        
        # Combine results
        combined_results = symbolic_results.copy()
        
        # Add neural predictions as symbolic facts
        for i, output_val in enumerate(neural_output[0]):
            if output_val > 0.5:
                neural_fact = f"neural_prediction_{i}(confidence_{output_val:.3f})"
                combined_results.append((neural_fact, float(output_val)))
        
        return combined_results, neural_output


class DifferentiableSymbolicReasoner:
    """
    Implements differentiable symbolic reasoning for end-to-end learning.
    """
    
    def __init__(self, max_reasoning_steps: int = 5):
        self.max_reasoning_steps = max_reasoning_steps
        self.reasoning_weights = np.random.randn(max_reasoning_steps) * 0.1
        self.rule_weights = defaultdict(lambda: np.random.randn() * 0.1)
        
    def differentiable_forward_chaining(self, 
                                      facts: Dict[str, float],
                                      rules: List[SymbolicRule]) -> Dict[str, float]:
        """Perform differentiable forward chaining reasoning."""
        current_facts = facts.copy()
        
        for step in range(self.max_reasoning_steps):
            new_facts = {}
            step_weight = self._sigmoid(self.reasoning_weights[step])
            
            for rule in rules:
                # Calculate rule activation
                premise_activations = []
                for premise in rule.premise:
                    if premise in current_facts:
                        premise_activations.append(current_facts[premise])
                    else:
                        premise_activations.append(0.0)
                
                if premise_activations:
                    # Soft AND operation (minimum)
                    rule_activation = min(premise_activations) * rule.confidence
                    rule_activation *= step_weight
                    rule_activation *= self._sigmoid(self.rule_weights[rule.conclusion])
                    
                    # Update conclusion fact
                    if rule.conclusion in new_facts:
                        new_facts[rule.conclusion] = max(new_facts[rule.conclusion], rule_activation)
                    else:
                        new_facts[rule.conclusion] = rule_activation
            
            # Merge new facts with current facts
            for fact, activation in new_facts.items():
                if fact in current_facts:
                    current_facts[fact] = max(current_facts[fact], activation)
                else:
                    current_facts[fact] = activation
        
        return current_facts
    
    def _sigmoid(self, x: float) -> float:
        """Sigmoid activation function."""
        return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))
    
    def update_weights(self, gradient_dict: Dict[str, float], learning_rate: float = 0.01):
        """Update reasoning weights based on gradients."""
        for i in range(len(self.reasoning_weights)):
            if f"step_{i}" in gradient_dict:
                self.reasoning_weights[i] -= learning_rate * gradient_dict[f"step_{i}"]
        
        for rule_conclusion, gradient in gradient_dict.items():
            if rule_conclusion.startswith("rule_"):
                self.rule_weights[rule_conclusion] -= learning_rate * gradient


class NeuroSymbolicIntegrator:
    """
    Main class for neural-symbolic integration in cognitive synergy.
    """
    
    def __init__(self, synergy_engine: CognitiveSynergyEngine):
        self.synergy_engine = synergy_engine
        self.neural_networks: Dict[str, NeuralSymbolicNetwork] = {}
        self.symbolic_reasoner = DifferentiableSymbolicReasoner()
        self.integration_history = []
        self.feature_mappings = {}
        
    def create_neural_symbolic_network(self, 
                                     network_id: str,
                                     input_size: int,
                                     hidden_size: int,
                                     output_size: int) -> NeuralSymbolicNetwork:
        """Create a new neural-symbolic network."""
        network = NeuralSymbolicNetwork(input_size, hidden_size, output_size)
        self.neural_networks[network_id] = network
        
        # Register as cognitive process
        process = CognitiveProcess(
            process_id=f"neural_symbolic_{network_id}",
            process_type="neural_symbolic",
            priority=0.8
        )
        self.synergy_engine.register_process(process)
        
        return network
    
    def extract_rules_from_atomspace(self) -> List[SymbolicRule]:
        """Extract symbolic rules from the AtomSpace."""
        rules = []
        memory = self.synergy_engine.memory
        
        # Get all atoms from memory
        if hasattr(memory.atoms, 'read'):
            atoms_dict = memory.atoms.read().unwrap()
        else:
            atoms_dict = memory.atoms
        
        # Look for implication links and similar structures
        for atom_id, atom in atoms_dict.items():
            if "ImplicationLink" in atom.atom_type or "Rule" in atom.atom_type:
                # Extract premise and conclusion from atom structure
                premises = []
                conclusion = ""
                
                # Get connected atoms
                for connected_id in atom.outgoing:
                    connected_atom = atoms_dict.get(connected_id)
                    if connected_atom:
                        if "Premise" in connected_atom.atom_type:
                            premises.append(connected_atom.name)
                        elif "Conclusion" in connected_atom.atom_type:
                            conclusion = connected_atom.name
                
                if premises and conclusion:
                    rule = SymbolicRule(
                        premise=premises,
                        conclusion=conclusion,
                        confidence=atom.truth_value,
                        extraction_method="atomspace_extraction"
                    )
                    rules.append(rule)
        
        return rules
    
    def inject_rules_to_atomspace(self, rules: List[SymbolicRule]):
        """Inject symbolic rules into the AtomSpace."""
        memory = self.synergy_engine.memory
        
        for i, rule in enumerate(rules):
            # Create rule atom
            rule_atom = Atom(
                atom_type="ImplicationLink",
                name=f"extracted_rule_{i}",
                truth_value=rule.confidence,
                attention_value=0.5
            )
            rule_id = memory.add_atom(rule_atom)
            
            # Create premise atoms
            for premise in rule.premise:
                premise_atom = Atom(
                    atom_type="PredicateNode",
                    name=premise,
                    truth_value=0.8,
                    attention_value=0.3
                )
                premise_id = memory.add_atom(premise_atom)
                memory.link_atoms(rule_id, premise_id, "premise")
            
            # Create conclusion atom
            conclusion_atom = Atom(
                atom_type="PredicateNode",
                name=rule.conclusion,
                truth_value=0.8,
                attention_value=0.4
            )
            conclusion_id = memory.add_atom(conclusion_atom)
            memory.link_atoms(rule_id, conclusion_id, "conclusion")
    
    def perform_integrated_reasoning(self, 
                                   query: str,
                                   neural_input: np.ndarray,
                                   network_id: str) -> Dict[str, Any]:
        """Perform integrated neural-symbolic reasoning."""
        if network_id not in self.neural_networks:
            raise ValueError(f"Neural network {network_id} not found")
        
        network = self.neural_networks[network_id]
        
        # Extract current symbolic rules
        atomspace_rules = self.extract_rules_from_atomspace()
        network_rules = network.symbolic_kb.rules
        all_rules = atomspace_rules + network_rules
        
        # Perform hybrid reasoning
        symbolic_results, neural_output = network.hybrid_reasoning(query, neural_input)
        
        # Use differentiable reasoner for additional inference
        initial_facts = {result[0]: result[1] for result in symbolic_results}
        reasoned_facts = self.symbolic_reasoner.differentiable_forward_chaining(
            initial_facts, all_rules
        )
        
        # Combine all results
        integrated_results = {
            'symbolic_results': symbolic_results,
            'neural_output': neural_output.tolist(),
            'reasoned_facts': reasoned_facts,
            'confidence': np.mean([conf for _, conf in symbolic_results]) if symbolic_results else 0.0,
            'integration_method': 'hybrid_neural_symbolic'
        }
        
        # Store integration history
        self.integration_history.append({
            'timestamp': datetime.now().isoformat(),
            'query': query,
            'results': integrated_results,
            'network_id': network_id
        })
        
        return integrated_results
    
    def learn_feature_mappings(self, 
                             symbolic_features: List[str],
                             neural_features: np.ndarray,
                             mapping_id: str):
        """Learn mappings between symbolic and neural features."""
        # Simple correlation-based mapping
        mappings = {}
        
        for i, sym_feature in enumerate(symbolic_features):
            if i < neural_features.shape[1]:
                # Calculate correlation or similarity
                correlation = np.corrcoef(
                    [hash(sym_feature) % 1000 / 1000.0] * neural_features.shape[0],
                    neural_features[:, i]
                )[0, 1]
                
                mappings[sym_feature] = {
                    'neural_index': i,
                    'correlation': correlation,
                    'mapping_strength': abs(correlation)
                }
        
        self.feature_mappings[mapping_id] = mappings
    
    def get_integration_summary(self) -> Dict[str, Any]:
        """Get summary of neural-symbolic integration performance."""
        if not self.integration_history:
            return {'status': 'no_integration_history'}
        
        # Calculate performance metrics
        confidences = [result['results']['confidence'] for result in self.integration_history]
        
        summary = {
            'total_integrations': len(self.integration_history),
            'average_confidence': np.mean(confidences),
            'confidence_std': np.std(confidences),
            'networks_used': len(self.neural_networks),
            'feature_mappings': len(self.feature_mappings),
            'recent_performance': confidences[-10:] if len(confidences) >= 10 else confidences
        }
        
        return summary


def demonstrate_neural_symbolic_integration():
    """
    Demonstrate neural-symbolic integration capabilities.
    """
    print("=== Neural-Symbolic Integration Demonstration ===\n")
    
    # Create synergy engine
    from cognitive_synergy_framework import CognitiveSynergyEngine
    engine = CognitiveSynergyEngine()
    
    # Create neural-symbolic integrator
    integrator = NeuroSymbolicIntegrator(engine)
    
    # Create a neural-symbolic network
    network = integrator.create_neural_symbolic_network(
        network_id="demo_network",
        input_size=4,
        hidden_size=8,
        output_size=2
    )
    
    print("Created neural-symbolic network:")
    print(f"  Input size: {network.input_size}")
    print(f"  Hidden size: {network.hidden_size}")
    print(f"  Output size: {network.output_size}")
    
    # Create some training data
    X_train = np.random.randn(100, 4)
    y_train = np.random.randint(0, 2, (100, 2)).astype(float)
    
    # Train the network
    print("\nTraining neural network...")
    for epoch in range(50):
        loss = network.train_step(X_train, y_train)
        if epoch % 10 == 0:
            print(f"  Epoch {epoch}, Loss: {loss:.4f}")
    
    # Extract symbolic rules from the trained network
    feature_names = ["feature_1", "feature_2", "feature_3", "feature_4"]
    output_names = ["class_A", "class_B"]
    
    print("\nExtracting symbolic rules from neural network...")
    extracted_rules = network.extract_symbolic_rules(
        X_train[:20], y_train[:20], feature_names, output_names
    )
    
    print(f"Extracted {len(extracted_rules)} symbolic rules:")
    for i, rule in enumerate(extracted_rules[:5]):  # Show first 5 rules
        print(f"  Rule {i+1}: {rule.to_logic_string()}")
    
    # Inject rules into AtomSpace
    print("\nInjecting rules into AtomSpace...")
    integrator.inject_rules_to_atomspace(extracted_rules)
    
    # Create some symbolic rules manually
    manual_rules = [
        SymbolicRule(
            premise=["feature_1(high)", "feature_2(high)"],
            conclusion="class_A(true)",
            confidence=0.9,
            extraction_method="manual"
        ),
        SymbolicRule(
            premise=["feature_3(low)", "feature_4(high)"],
            conclusion="class_B(true)",
            confidence=0.8,
            extraction_method="manual"
        )
    ]
    
    # Inject symbolic knowledge into neural network
    print("\nInjecting symbolic knowledge into neural network...")
    network.inject_symbolic_knowledge(manual_rules)
    
    # Perform integrated reasoning
    print("\nPerforming integrated neural-symbolic reasoning...")
    test_input = np.array([0.8, 0.7, -0.3, 0.9])
    query = "class_A(true)"
    
    results = integrator.perform_integrated_reasoning(
        query=query,
        neural_input=test_input,
        network_id="demo_network"
    )
    
    print(f"Query: {query}")
    print(f"Neural input: {test_input}")
    print(f"Results:")
    print(f"  Symbolic results: {results['symbolic_results']}")
    print(f"  Neural output: {results['neural_output']}")
    print(f"  Reasoned facts: {list(results['reasoned_facts'].items())[:5]}")
    print(f"  Overall confidence: {results['confidence']:.3f}")
    
    # Learn feature mappings
    print("\nLearning feature mappings...")
    integrator.learn_feature_mappings(
        symbolic_features=feature_names,
        neural_features=X_train,
        mapping_id="demo_mapping"
    )
    
    # Get integration summary
    print("\nIntegration Summary:")
    summary = integrator.get_integration_summary()
    for key, value in summary.items():
        if isinstance(value, (int, float)):
            print(f"  {key}: {value:.3f}" if isinstance(value, float) else f"  {key}: {value}")
        elif isinstance(value, list) and len(value) <= 5:
            print(f"  {key}: {value}")
    
    print("\n=== Neural-Symbolic Integration Demo Complete ===")


if __name__ == "__main__":
    demonstrate_neural_symbolic_integration()

// Cognitive Synergy Implementation in Rust
// ========================================
//
// This module implements core cognitive synergy concepts in Rust,
// providing high-performance components for the OpenCog Collection.
//
// Key features:
// - Hypergraph memory structures
// - Attention allocation mechanisms
// - Pattern mining algorithms
// - Inter-component communication
// - Cognitive process coordination

use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::{Arc, Mutex, RwLock};
use std::thread;
use std::time::{Duration, Instant};
use serde::{Deserialize, Serialize};

/// Represents an atom in the hypergraph memory
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Atom {
    pub atom_type: String,
    pub name: String,
    pub truth_value: f64,
    pub attention_value: f64,
    pub incoming: HashSet<String>,
    pub outgoing: HashSet<String>,
    pub metadata: HashMap<String, String>,
}

impl Atom {
    pub fn new(atom_type: String, name: String) -> Self {
        Self {
            atom_type,
            name,
            truth_value: 1.0,
            attention_value: 0.0,
            incoming: HashSet::new(),
            outgoing: HashSet::new(),
            metadata: HashMap::new(),
        }
    }
    
    pub fn with_truth_value(mut self, truth_value: f64) -> Self {
        self.truth_value = truth_value;
        self
    }
    
    pub fn with_attention(mut self, attention_value: f64) -> Self {
        self.attention_value = attention_value;
        self
    }
    
    pub fn get_id(&self) -> String {
        format!("{}:{}", self.atom_type, self.name)
    }
}

/// Hypergraph memory for storing and managing atoms
#[derive(Debug)]
pub struct HypergraphMemory {
    atoms: Arc<RwLock<HashMap<String, Atom>>>,
    attention_bank: Arc<RwLock<HashMap<String, f64>>>,
    pattern_cache: Arc<RwLock<HashMap<String, Vec<String>>>>,
}

impl HypergraphMemory {
    pub fn new() -> Self {
        Self {
            atoms: Arc::new(RwLock::new(HashMap::new())),
            attention_bank: Arc::new(RwLock::new(HashMap::new())),
            pattern_cache: Arc::new(RwLock::new(HashMap::new())),
        }
    }
    
    pub fn add_atom(&self, atom: Atom) -> String {
        let atom_id = atom.get_id();
        let attention_value = atom.attention_value;
        
        {
            let mut atoms = self.atoms.write().unwrap();
            atoms.insert(atom_id.clone(), atom);
        }
        
        {
            let mut attention_bank = self.attention_bank.write().unwrap();
            attention_bank.insert(atom_id.clone(), attention_value);
        }
        
        atom_id
    }
    
    pub fn get_atom(&self, atom_id: &str) -> Option<Atom> {
        let atoms = self.atoms.read().unwrap();
        atoms.get(atom_id).cloned()
    }
    
    pub fn link_atoms(&self, source_id: &str, target_id: &str, link_type: &str) {
        let mut atoms = self.atoms.write().unwrap();
        
        // Check if both atoms exist
        let source_exists = atoms.contains_key(source_id);
        let target_exists = atoms.contains_key(target_id);
        
        if source_exists && target_exists {
            // Update source atom
            if let Some(source_atom) = atoms.get_mut(source_id) {
                source_atom.outgoing.insert(target_id.to_string());
            }
            
            // Update target atom
            if let Some(target_atom) = atoms.get_mut(target_id) {
                target_atom.incoming.insert(source_id.to_string());
            }
            
            // Create link atom
            let link_atom = Atom::new(
                format!("Link:{}", link_type),
                format!("{}->{}", source_id, target_id),
            );
            
            let link_id = link_atom.get_id();
            atoms.insert(link_id, link_atom);
        }
    }
    
    pub fn update_attention(&self, atom_id: &str, attention_delta: f64) {
        let current_attention = {
            let attention_bank = self.attention_bank.read().unwrap();
            *attention_bank.get(atom_id).unwrap_or(&0.0)
        };
        
        let new_attention = current_attention + attention_delta;
        
        {
            let mut attention_bank = self.attention_bank.write().unwrap();
            attention_bank.insert(atom_id.to_string(), new_attention);
        }
        
        // Update atom's attention value
        if let Ok(mut atoms) = self.atoms.write() {
            if let Some(atom) = atoms.get_mut(atom_id) {
                atom.attention_value = new_attention;
            }
        }
    }
    
    pub fn get_high_attention_atoms(&self, threshold: f64) -> Vec<String> {
        let attention_bank = self.attention_bank.read().unwrap();
        attention_bank
            .iter()
            .filter(|(_, &attention)| attention >= threshold)
            .map(|(atom_id, _)| atom_id.clone())
            .collect()
    }
    
    pub fn get_neighbors(&self, atom_id: &str) -> HashSet<String> {
        let atoms = self.atoms.read().unwrap();
        if let Some(atom) = atoms.get(atom_id) {
            let mut neighbors = atom.incoming.clone();
            neighbors.extend(atom.outgoing.clone());
            neighbors
        } else {
            HashSet::new()
        }
    }
    
    pub fn atom_count(&self) -> usize {
        let atoms = self.atoms.read().unwrap();
        atoms.len()
    }
}

/// Represents a cognitive process that can participate in synergy
#[derive(Debug, Clone)]
pub struct CognitiveProcess {
    pub process_id: String,
    pub process_type: String,
    pub priority: f64,
    pub bottleneck_threshold: f64,
    pub is_stuck: bool,
    pub performance_metrics: HashMap<String, f64>,
}

impl CognitiveProcess {
    pub fn new(process_id: String, process_type: String) -> Self {
        Self {
            process_id,
            process_type,
            priority: 1.0,
            bottleneck_threshold: 0.8,
            is_stuck: false,
            performance_metrics: HashMap::new(),
        }
    }
    
    pub fn with_priority(mut self, priority: f64) -> Self {
        self.priority = priority;
        self
    }
    
    pub fn update_performance(&mut self, metric: String, value: f64) {
        self.performance_metrics.insert(metric, value);
    }
    
    pub fn average_performance(&self) -> f64 {
        if self.performance_metrics.is_empty() {
            0.5
        } else {
            let sum: f64 = self.performance_metrics.values().sum();
            sum / self.performance_metrics.len() as f64
        }
    }
}

/// Pattern miner for discovering cognitive synergy patterns
#[derive(Debug)]
pub struct PatternMiner {
    pub miner_id: String,
    discovered_patterns: Vec<Pattern>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Pattern {
    pub pattern_type: String,
    pub atoms: Vec<String>,
    pub frequency: usize,
    pub confidence: f64,
    pub discovered_by: String,
}

impl PatternMiner {
    pub fn new(miner_id: String) -> Self {
        Self {
            miner_id,
            discovered_patterns: Vec::new(),
        }
    }
    
    pub fn mine_patterns(&mut self, memory: &HypergraphMemory) -> Vec<Pattern> {
        let mut patterns = Vec::new();
        let atoms = memory.atoms.read().unwrap();
        
        // Mine frequent connection patterns
        let mut connection_counts: HashMap<(String, String), usize> = HashMap::new();
        
        for (atom_id, atom) in atoms.iter() {
            for neighbor_id in &atom.outgoing {
                let mut pair = vec![atom_id.clone(), neighbor_id.clone()];
                pair.sort();
                let key = (pair[0].clone(), pair[1].clone());
                *connection_counts.entry(key).or_insert(0) += 1;
            }
        }
        
        // Extract significant patterns
        for ((atom1, atom2), count) in connection_counts {
            if count > 1 {  // Threshold for pattern significance
                let pattern = Pattern {
                    pattern_type: "frequent_connection".to_string(),
                    atoms: vec![atom1, atom2],
                    frequency: count,
                    confidence: count as f64 / atoms.len() as f64,
                    discovered_by: self.miner_id.clone(),
                };
                patterns.push(pattern);
            }
        }
        
        self.discovered_patterns.extend(patterns.clone());
        patterns
    }
    
    pub fn get_discovered_patterns(&self) -> &Vec<Pattern> {
        &self.discovered_patterns
    }
}

/// Attention allocator for managing cognitive attention
#[derive(Debug)]
pub struct AttentionAllocator {
    attention_budget: f64,
    decay_rate: f64,
}

impl AttentionAllocator {
    pub fn new() -> Self {
        Self {
            attention_budget: 100.0,
            decay_rate: 0.95,
        }
    }
    
    pub fn allocate_attention(&self, memory: &HypergraphMemory) {
        let high_attention_atoms = memory.get_high_attention_atoms(0.3);
        
        for atom_id in high_attention_atoms {
            let neighbors = memory.get_neighbors(&atom_id);
            let attention_per_neighbor = if neighbors.is_empty() {
                0.0
            } else {
                let current_attention = {
                    let attention_bank = memory.attention_bank.read().unwrap();
                    *attention_bank.get(&atom_id).unwrap_or(&0.0)
                };
                current_attention * 0.1 / neighbors.len() as f64
            };
            
            for neighbor_id in neighbors {
                memory.update_attention(&neighbor_id, attention_per_neighbor);
            }
        }
        
        // Apply attention decay
        let attention_bank = memory.attention_bank.read().unwrap();
        let atom_ids: Vec<String> = attention_bank.keys().cloned().collect();
        drop(attention_bank);
        
        for atom_id in atom_ids {
            let current_attention = {
                let attention_bank = memory.attention_bank.read().unwrap();
                *attention_bank.get(&atom_id).unwrap_or(&0.0)
            };
            let new_attention = current_attention * self.decay_rate;
            memory.update_attention(&atom_id, new_attention - current_attention);
        }
    }
}

/// Main cognitive synergy engine
#[derive(Debug)]
pub struct CognitiveSynergyEngine {
    memory: Arc<HypergraphMemory>,
    processes: Arc<RwLock<HashMap<String, CognitiveProcess>>>,
    pattern_miners: Arc<RwLock<Vec<PatternMiner>>>,
    attention_allocator: AttentionAllocator,
    running: Arc<Mutex<bool>>,
    synergy_metrics: Arc<RwLock<HashMap<String, f64>>>,
}

impl CognitiveSynergyEngine {
    pub fn new() -> Self {
        Self {
            memory: Arc::new(HypergraphMemory::new()),
            processes: Arc::new(RwLock::new(HashMap::new())),
            pattern_miners: Arc::new(RwLock::new(Vec::new())),
            attention_allocator: AttentionAllocator::new(),
            running: Arc::new(Mutex::new(false)),
            synergy_metrics: Arc::new(RwLock::new(HashMap::new())),
        }
    }
    
    pub fn register_process(&self, process: CognitiveProcess) {
        let mut processes = self.processes.write().unwrap();
        processes.insert(process.process_id.clone(), process);
    }
    
    pub fn add_pattern_miner(&self, miner: PatternMiner) {
        let mut miners = self.pattern_miners.write().unwrap();
        miners.push(miner);
    }
    
    pub fn get_memory(&self) -> Arc<HypergraphMemory> {
        self.memory.clone()
    }
    
    pub fn start_synergy_loop(&self) {
        {
            let mut running = self.running.lock().unwrap();
            *running = true;
        }
        
        let memory = self.memory.clone();
        let processes = self.processes.clone();
        let pattern_miners = self.pattern_miners.clone();
        let running = self.running.clone();
        let synergy_metrics = self.synergy_metrics.clone();
        
        thread::spawn(move || {
            while *running.lock().unwrap() {
                // Allocate attention
                let allocator = AttentionAllocator::new();
                allocator.allocate_attention(&memory);
                
                // Mine patterns
                {
                    let mut miners = pattern_miners.write().unwrap();
                    for miner in miners.iter_mut() {
                        let _patterns = miner.mine_patterns(&memory);
                    }
                }
                
                // Detect bottlenecks
                let _stuck_processes = {
                    let mut processes_guard = processes.write().unwrap();
                    let mut stuck = Vec::new();
                    
                    for (process_id, process) in processes_guard.iter_mut() {
                        if process.average_performance() < process.bottleneck_threshold {
                            process.is_stuck = true;
                            stuck.push(process_id.clone());
                        } else {
                            process.is_stuck = false;
                        }
                    }
                    stuck
                };
                
                // Update synergy metrics
                {
                    let mut metrics = synergy_metrics.write().unwrap();
                    let processes_guard = processes.read().unwrap();
                    let total_processes = processes_guard.len() as f64;
                    let active_processes = processes_guard
                        .values()
                        .filter(|p| !p.is_stuck)
                        .count() as f64;
                    
                    if total_processes > 0.0 {
                        metrics.insert("process_efficiency".to_string(), active_processes / total_processes);
                    }
                    
                    metrics.insert("attention_distribution".to_string(), 
                                 memory.get_high_attention_atoms(0.3).len() as f64);
                    metrics.insert("total_atoms".to_string(), memory.atom_count() as f64);
                }
                
                // Sleep before next iteration
                thread::sleep(Duration::from_millis(100));
            }
        });
    }
    
    pub fn stop_synergy_loop(&self) {
        let mut running = self.running.lock().unwrap();
        *running = false;
    }
    
    pub fn get_synergy_metrics(&self) -> HashMap<String, f64> {
        let metrics = self.synergy_metrics.read().unwrap();
        metrics.clone()
    }
}

/// Demonstration function for the cognitive synergy system
pub fn demonstrate_cognitive_synergy() {
    println!("=== Rust Cognitive Synergy Demonstration ===");
    
    // Create synergy engine
    let engine = CognitiveSynergyEngine::new();
    let memory = engine.get_memory();
    
    // Register cognitive processes
    let reasoning_process = CognitiveProcess::new(
        "reasoning_engine".to_string(),
        "symbolic_reasoning".to_string(),
    ).with_priority(0.8);
    
    let learning_process = CognitiveProcess::new(
        "pattern_learner".to_string(),
        "machine_learning".to_string(),
    ).with_priority(0.7);
    
    engine.register_process(reasoning_process);
    engine.register_process(learning_process);
    
    // Add pattern miner
    let miner = PatternMiner::new("rust_miner".to_string());
    engine.add_pattern_miner(miner);
    
    // Create some example atoms
    let concept_atom = Atom::new("ConceptNode".to_string(), "intelligence".to_string())
        .with_attention(0.8);
    let property_atom = Atom::new("PredicateNode".to_string(), "has_property".to_string())
        .with_attention(0.6);
    let value_atom = Atom::new("ConceptNode".to_string(), "emergent".to_string())
        .with_attention(0.7);
    
    // Add atoms to memory
    let concept_id = memory.add_atom(concept_atom);
    let property_id = memory.add_atom(property_atom);
    let value_id = memory.add_atom(value_atom);
    
    // Create links
    memory.link_atoms(&concept_id, &property_id, "evaluation");
    memory.link_atoms(&property_id, &value_id, "evaluation");
    
    println!("Initial memory state:");
    println!("  Atoms in memory: {}", memory.atom_count());
    println!("  High attention atoms: {:?}", memory.get_high_attention_atoms(0.5));
    
    // Start synergy loop
    engine.start_synergy_loop();
    
    // Let it run for a short time
    thread::sleep(Duration::from_millis(500));
    
    // Check results
    println!("\nAfter synergy processing:");
    println!("  High attention atoms: {:?}", memory.get_high_attention_atoms(0.3));
    println!("  Synergy metrics: {:?}", engine.get_synergy_metrics());
    
    // Stop the engine
    engine.stop_synergy_loop();
    
    println!("\n=== Demonstration Complete ===");
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_atom_creation() {
        let atom = Atom::new("ConceptNode".to_string(), "test".to_string())
            .with_truth_value(0.9)
            .with_attention(0.5);
        
        assert_eq!(atom.atom_type, "ConceptNode");
        assert_eq!(atom.name, "test");
        assert_eq!(atom.truth_value, 0.9);
        assert_eq!(atom.attention_value, 0.5);
    }
    
    #[test]
    fn test_hypergraph_memory() {
        let memory = HypergraphMemory::new();
        let atom = Atom::new("ConceptNode".to_string(), "test".to_string());
        let atom_id = memory.add_atom(atom);
        
        assert!(memory.get_atom(&atom_id).is_some());
        assert_eq!(memory.atom_count(), 1);
    }
    
    #[test]
    fn test_pattern_mining() {
        let memory = HypergraphMemory::new();
        let mut miner = PatternMiner::new("test_miner".to_string());
        
        // Add some atoms and links
        let atom1 = Atom::new("ConceptNode".to_string(), "atom1".to_string());
        let atom2 = Atom::new("ConceptNode".to_string(), "atom2".to_string());
        
        let id1 = memory.add_atom(atom1);
        let id2 = memory.add_atom(atom2);
        memory.link_atoms(&id1, &id2, "test_link");
        
        let patterns = miner.mine_patterns(&memory);
        assert!(!patterns.is_empty());
    }
}

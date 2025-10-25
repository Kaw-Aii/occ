"""
Deep Tree Echo Membrane Architecture
====================================

This module implements the hierarchical membrane architecture for organizing
cognitive processes in the OpenCog Collection, inspired by P-systems and
membrane computing.

The membrane hierarchy provides:
- Hierarchical organization of cognitive processes
- Isolation and security boundaries
- Communication protocols between membranes
- Resource allocation and attention management
- Fault tolerance and recovery

Membrane Hierarchy:
- Root Membrane (System Boundary)
  - Cognitive Membrane (Core Processing)
    - Memory Membrane (Storage & Retrieval)
    - Reasoning Membrane (Inference & Logic)
    - Grammar Membrane (Symbolic Processing)
  - Extension Membrane (Plugin Container)
    - Browser Membrane
    - ML Membrane
    - Introspection Membrane
  - Security Membrane (Validation & Control)
    - Authentication Membrane
    - Validation Membrane
    - Emergency Membrane

Author: OpenCog Collection Contributors
License: GPL-3.0+
"""

import logging
from typing import Dict, List, Any, Optional, Set, Callable
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
from collections import defaultdict, deque
import threading
from queue import Queue, PriorityQueue
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MembraneType(Enum):
    """Types of membranes in the hierarchy."""
    ROOT = "root"
    COGNITIVE = "cognitive"
    MEMORY = "memory"
    REASONING = "reasoning"
    GRAMMAR = "grammar"
    EXTENSION = "extension"
    BROWSER = "browser"
    ML = "ml"
    INTROSPECTION = "introspection"
    SECURITY = "security"
    AUTHENTICATION = "authentication"
    VALIDATION = "validation"
    EMERGENCY = "emergency"


class MessagePriority(Enum):
    """Priority levels for inter-membrane messages."""
    CRITICAL = 0
    HIGH = 1
    NORMAL = 2
    LOW = 3


@dataclass
class MembraneMessage:
    """Message passed between membranes."""
    source_membrane: str
    target_membrane: str
    message_type: str
    payload: Any
    priority: MessagePriority = MessagePriority.NORMAL
    timestamp: datetime = field(default_factory=datetime.now)
    message_id: str = field(default_factory=lambda: f"msg_{datetime.now().timestamp()}")


@dataclass
class MembraneState:
    """State of a membrane."""
    active: bool = True
    resource_usage: float = 0.0
    message_count: int = 0
    error_count: int = 0
    last_activity: datetime = field(default_factory=datetime.now)
    performance_metrics: Dict[str, float] = field(default_factory=dict)


class Membrane:
    """
    Base class for all membranes in the hierarchy.
    Provides communication, isolation, and resource management.
    """
    
    def __init__(self, membrane_id: str, membrane_type: MembraneType,
                 parent: Optional['Membrane'] = None):
        self.membrane_id = membrane_id
        self.membrane_type = membrane_type
        self.parent = parent
        self.children: Dict[str, 'Membrane'] = {}
        
        self.state = MembraneState()
        self.message_queue = PriorityQueue()
        self.message_handlers: Dict[str, Callable] = {}
        
        self.lock = threading.RLock()
        self.running = False
        self.worker_thread: Optional[threading.Thread] = None
        
        logger.info(f"Membrane created: {membrane_id} ({membrane_type.value})")
    
    def add_child(self, child: 'Membrane'):
        """Add a child membrane."""
        with self.lock:
            self.children[child.membrane_id] = child
            child.parent = self
            logger.debug(f"Child membrane added: {child.membrane_id} -> {self.membrane_id}")
    
    def remove_child(self, child_id: str):
        """Remove a child membrane."""
        with self.lock:
            if child_id in self.children:
                del self.children[child_id]
                logger.debug(f"Child membrane removed: {child_id}")
    
    def send_message(self, target_id: str, message_type: str, payload: Any,
                    priority: MessagePriority = MessagePriority.NORMAL):
        """Send a message to another membrane."""
        message = MembraneMessage(
            source_membrane=self.membrane_id,
            target_membrane=target_id,
            message_type=message_type,
            payload=payload,
            priority=priority
        )
        
        # Find target membrane
        target = self._find_membrane(target_id)
        if target:
            target.receive_message(message)
            logger.debug(f"Message sent: {self.membrane_id} -> {target_id} ({message_type})")
        else:
            logger.warning(f"Target membrane not found: {target_id}")
    
    def receive_message(self, message: MembraneMessage):
        """Receive a message from another membrane."""
        with self.lock:
            self.message_queue.put((message.priority.value, message))
            self.state.message_count += 1
    
    def register_handler(self, message_type: str, handler: Callable):
        """Register a handler for a specific message type."""
        self.message_handlers[message_type] = handler
        logger.debug(f"Handler registered: {message_type} in {self.membrane_id}")
    
    def process_messages(self):
        """Process messages in the queue."""
        while not self.message_queue.empty():
            try:
                _, message = self.message_queue.get_nowait()
                
                # Find and execute handler
                if message.message_type in self.message_handlers:
                    handler = self.message_handlers[message.message_type]
                    handler(message)
                else:
                    self._default_handler(message)
                
                self.state.last_activity = datetime.now()
                
            except Exception as e:
                logger.error(f"Error processing message in {self.membrane_id}: {e}")
                self.state.error_count += 1
    
    def _default_handler(self, message: MembraneMessage):
        """Default message handler."""
        logger.debug(f"Default handler: {message.message_type} in {self.membrane_id}")
    
    def _find_membrane(self, membrane_id: str) -> Optional['Membrane']:
        """Find a membrane by ID in the hierarchy."""
        # Check self
        if self.membrane_id == membrane_id:
            return self
        
        # Check children
        if membrane_id in self.children:
            return self.children[membrane_id]
        
        # Check parent
        if self.parent:
            return self.parent._find_membrane(membrane_id)
        
        # Recursively check children
        for child in self.children.values():
            result = child._find_membrane(membrane_id)
            if result:
                return result
        
        return None
    
    def start(self):
        """Start the membrane's message processing loop."""
        if not self.running:
            self.running = True
            self.worker_thread = threading.Thread(target=self._run_loop, daemon=True)
            self.worker_thread.start()
            logger.info(f"Membrane started: {self.membrane_id}")
    
    def stop(self):
        """Stop the membrane's message processing loop."""
        self.running = False
        if self.worker_thread:
            self.worker_thread.join(timeout=5.0)
        logger.info(f"Membrane stopped: {self.membrane_id}")
    
    def _run_loop(self):
        """Main processing loop for the membrane."""
        while self.running:
            try:
                self.process_messages()
                self._update_metrics()
                threading.Event().wait(0.1)  # Small delay
            except Exception as e:
                logger.error(f"Error in membrane loop {self.membrane_id}: {e}")
    
    def _update_metrics(self):
        """Update performance metrics."""
        self.state.performance_metrics['message_count'] = self.state.message_count
        self.state.performance_metrics['error_count'] = self.state.error_count
        self.state.performance_metrics['queue_size'] = self.message_queue.qsize()
    
    def get_status(self) -> Dict[str, Any]:
        """Get current status of the membrane."""
        return {
            'membrane_id': self.membrane_id,
            'type': self.membrane_type.value,
            'active': self.state.active,
            'running': self.running,
            'message_count': self.state.message_count,
            'error_count': self.state.error_count,
            'queue_size': self.message_queue.qsize(),
            'children': list(self.children.keys()),
            'last_activity': self.state.last_activity.isoformat()
        }


class CognitiveMembrane(Membrane):
    """Cognitive membrane for core processing."""
    
    def __init__(self, membrane_id: str = "cognitive", parent: Optional[Membrane] = None):
        super().__init__(membrane_id, MembraneType.COGNITIVE, parent)
        
        # Register handlers for cognitive operations
        self.register_handler("process_pattern", self._handle_pattern)
        self.register_handler("reason", self._handle_reasoning)
        self.register_handler("learn", self._handle_learning)
    
    def _handle_pattern(self, message: MembraneMessage):
        """Handle pattern processing."""
        logger.info(f"Processing pattern: {message.payload}")
        # Pattern processing logic here
    
    def _handle_reasoning(self, message: MembraneMessage):
        """Handle reasoning request."""
        logger.info(f"Reasoning: {message.payload}")
        # Reasoning logic here
    
    def _handle_learning(self, message: MembraneMessage):
        """Handle learning request."""
        logger.info(f"Learning: {message.payload}")
        # Learning logic here


class MemoryMembrane(Membrane):
    """Memory membrane for storage and retrieval."""
    
    def __init__(self, membrane_id: str = "memory", parent: Optional[Membrane] = None):
        super().__init__(membrane_id, MembraneType.MEMORY, parent)
        
        self.memory_store: Dict[str, Any] = {}
        
        # Register handlers
        self.register_handler("store", self._handle_store)
        self.register_handler("retrieve", self._handle_retrieve)
        self.register_handler("query", self._handle_query)
    
    def _handle_store(self, message: MembraneMessage):
        """Store data in memory."""
        key = message.payload.get('key')
        value = message.payload.get('value')
        if key:
            self.memory_store[key] = value
            logger.debug(f"Stored: {key}")
    
    def _handle_retrieve(self, message: MembraneMessage):
        """Retrieve data from memory."""
        key = message.payload.get('key')
        value = self.memory_store.get(key)
        
        # Send response back to source
        self.send_message(
            message.source_membrane,
            "retrieve_response",
            {'key': key, 'value': value}
        )
    
    def _handle_query(self, message: MembraneMessage):
        """Query memory with pattern."""
        pattern = message.payload.get('pattern', '')
        results = {k: v for k, v in self.memory_store.items() if pattern in k}
        
        self.send_message(
            message.source_membrane,
            "query_response",
            {'results': results}
        )


class SecurityMembrane(Membrane):
    """Security membrane for validation and control."""
    
    def __init__(self, membrane_id: str = "security", parent: Optional[Membrane] = None):
        super().__init__(membrane_id, MembraneType.SECURITY, parent)
        
        self.access_log: List[Dict[str, Any]] = []
        self.blocked_operations: Set[str] = set()
        
        # Register handlers
        self.register_handler("validate", self._handle_validate)
        self.register_handler("authorize", self._handle_authorize)
        self.register_handler("emergency", self._handle_emergency)
    
    def _handle_validate(self, message: MembraneMessage):
        """Validate an operation."""
        operation = message.payload.get('operation')
        is_valid = operation not in self.blocked_operations
        
        self.access_log.append({
            'timestamp': datetime.now().isoformat(),
            'source': message.source_membrane,
            'operation': operation,
            'valid': is_valid
        })
        
        self.send_message(
            message.source_membrane,
            "validate_response",
            {'valid': is_valid}
        )
    
    def _handle_authorize(self, message: MembraneMessage):
        """Authorize access."""
        logger.info(f"Authorization request: {message.payload}")
    
    def _handle_emergency(self, message: MembraneMessage):
        """Handle emergency shutdown."""
        logger.critical(f"EMERGENCY: {message.payload}")
        # Emergency procedures here


class DeepTreeEchoArchitecture:
    """
    Complete Deep Tree Echo membrane architecture.
    Manages the entire membrane hierarchy.
    """
    
    def __init__(self):
        # Create root membrane
        self.root = Membrane("root", MembraneType.ROOT)
        
        # Create cognitive membrane hierarchy
        self.cognitive = CognitiveMembrane("cognitive", self.root)
        self.root.add_child(self.cognitive)
        
        self.memory = MemoryMembrane("memory", self.cognitive)
        self.cognitive.add_child(self.memory)
        
        self.reasoning = Membrane("reasoning", MembraneType.REASONING, self.cognitive)
        self.cognitive.add_child(self.reasoning)
        
        self.grammar = Membrane("grammar", MembraneType.GRAMMAR, self.cognitive)
        self.cognitive.add_child(self.grammar)
        
        # Create extension membrane hierarchy
        self.extension = Membrane("extension", MembraneType.EXTENSION, self.root)
        self.root.add_child(self.extension)
        
        self.browser = Membrane("browser", MembraneType.BROWSER, self.extension)
        self.extension.add_child(self.browser)
        
        self.ml = Membrane("ml", MembraneType.ML, self.extension)
        self.extension.add_child(self.ml)
        
        self.introspection = Membrane("introspection", MembraneType.INTROSPECTION, self.extension)
        self.extension.add_child(self.introspection)
        
        # Create security membrane hierarchy
        self.security = SecurityMembrane("security", self.root)
        self.root.add_child(self.security)
        
        self.authentication = Membrane("authentication", MembraneType.AUTHENTICATION, self.security)
        self.security.add_child(self.authentication)
        
        self.validation = Membrane("validation", MembraneType.VALIDATION, self.security)
        self.security.add_child(self.validation)
        
        self.emergency = Membrane("emergency", MembraneType.EMERGENCY, self.security)
        self.security.add_child(self.emergency)
        
        logger.info("Deep Tree Echo Architecture initialized")
    
    def start_all(self):
        """Start all membranes."""
        self._start_recursive(self.root)
        logger.info("All membranes started")
    
    def stop_all(self):
        """Stop all membranes."""
        self._stop_recursive(self.root)
        logger.info("All membranes stopped")
    
    def _start_recursive(self, membrane: Membrane):
        """Recursively start membranes."""
        membrane.start()
        for child in membrane.children.values():
            self._start_recursive(child)
    
    def _stop_recursive(self, membrane: Membrane):
        """Recursively stop membranes."""
        for child in membrane.children.values():
            self._stop_recursive(child)
        membrane.stop()
    
    def get_hierarchy_status(self) -> Dict[str, Any]:
        """Get status of entire membrane hierarchy."""
        return self._get_status_recursive(self.root)
    
    def _get_status_recursive(self, membrane: Membrane) -> Dict[str, Any]:
        """Recursively get status."""
        status = membrane.get_status()
        status['children_status'] = {
            child_id: self._get_status_recursive(child)
            for child_id, child in membrane.children.items()
        }
        return status
    
    def export_architecture(self, filepath: str):
        """Export architecture status to file."""
        status = self.get_hierarchy_status()
        with open(filepath, 'w') as f:
            json.dump(status, f, indent=2)
        logger.info(f"Architecture exported to {filepath}")


# Example usage
if __name__ == "__main__":
    print("=== Deep Tree Echo Membrane Architecture ===\n")
    
    # Create architecture
    arch = DeepTreeEchoArchitecture()
    
    # Start all membranes
    arch.start_all()
    
    # Send some test messages
    arch.memory.send_message(
        "memory",
        "store",
        {'key': 'test_pattern', 'value': {'type': 'ConceptNode', 'name': 'AGI'}}
    )
    
    arch.cognitive.send_message(
        "memory",
        "retrieve",
        {'key': 'test_pattern'}
    )
    
    # Wait for processing
    import time
    time.sleep(1)
    
    # Get status
    status = arch.get_hierarchy_status()
    print("Architecture Status:")
    print(json.dumps(status, indent=2))
    
    # Export architecture
    arch.export_architecture('/tmp/deep_tree_echo_architecture.json')
    print("\nArchitecture exported to /tmp/deep_tree_echo_architecture.json")
    
    # Stop all membranes
    arch.stop_all()


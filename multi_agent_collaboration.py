#!/usr/bin/env python3
"""
Multi-Agent Cognitive Collaboration System for OpenCog Collection
================================================================

This module implements advanced multi-agent cognitive collaboration capabilities,
enabling multiple AI agents to work together synergistically on complex problems
through distributed cognition, knowledge sharing, and collaborative reasoning.

Key Features:
- Distributed cognitive processing across multiple agents
- Dynamic task allocation and load balancing
- Inter-agent communication and knowledge sharing
- Collaborative problem-solving strategies
- Emergent collective intelligence
- Fault tolerance and redundancy

Author: OpenCog Collection Contributors
License: GPL-3.0+
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Set, Union, Callable
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import json
import logging
from datetime import datetime, timedelta
import threading
import asyncio
import queue
from collections import defaultdict, deque
from enum import Enum
import uuid
import time
import statistics

from cognitive_synergy_framework import (
    CognitiveSynergyEngine, CognitiveProcess, Atom, 
    HypergraphMemory, PatternMiner
)

logger = logging.getLogger(__name__)


class AgentRole(Enum):
    """Roles that agents can take in collaborative tasks."""
    COORDINATOR = "coordinator"
    SPECIALIST = "specialist"
    GENERALIST = "generalist"
    VALIDATOR = "validator"
    SYNTHESIZER = "synthesizer"
    EXPLORER = "explorer"
    OPTIMIZER = "optimizer"


class TaskStatus(Enum):
    """Status of collaborative tasks."""
    PENDING = "pending"
    ASSIGNED = "assigned"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class MessageType(Enum):
    """Types of inter-agent messages."""
    TASK_REQUEST = "task_request"
    TASK_RESPONSE = "task_response"
    KNOWLEDGE_SHARE = "knowledge_share"
    COORDINATION = "coordination"
    STATUS_UPDATE = "status_update"
    HELP_REQUEST = "help_request"
    RESULT_SHARE = "result_share"


@dataclass
class AgentMessage:
    """Message structure for inter-agent communication."""
    message_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    sender_id: str = ""
    receiver_id: str = ""
    message_type: MessageType = MessageType.COORDINATION
    content: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    priority: float = 0.5
    requires_response: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert message to dictionary."""
        return {
            'message_id': self.message_id,
            'sender_id': self.sender_id,
            'receiver_id': self.receiver_id,
            'message_type': self.message_type.value,
            'content': self.content,
            'timestamp': self.timestamp.isoformat(),
            'priority': self.priority,
            'requires_response': self.requires_response
        }


@dataclass
class CollaborativeTask:
    """Represents a task that can be solved collaboratively."""
    task_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    description: str = ""
    requirements: List[str] = field(default_factory=list)
    priority: float = 0.5
    deadline: Optional[datetime] = None
    assigned_agents: List[str] = field(default_factory=list)
    status: TaskStatus = TaskStatus.PENDING
    subtasks: List['CollaborativeTask'] = field(default_factory=list)
    results: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None
    
    def add_subtask(self, subtask: 'CollaborativeTask'):
        """Add a subtask to this task."""
        self.subtasks.append(subtask)
    
    def is_complete(self) -> bool:
        """Check if task and all subtasks are complete."""
        if self.status != TaskStatus.COMPLETED:
            return False
        return all(subtask.is_complete() for subtask in self.subtasks)
    
    def get_progress(self) -> float:
        """Get overall progress of task including subtasks."""
        if self.status == TaskStatus.COMPLETED:
            return 1.0
        elif self.status in [TaskStatus.FAILED, TaskStatus.CANCELLED]:
            return 0.0
        
        if not self.subtasks:
            return 0.5 if self.status == TaskStatus.IN_PROGRESS else 0.0
        
        subtask_progress = [subtask.get_progress() for subtask in self.subtasks]
        return statistics.mean(subtask_progress)


class CognitiveAgent:
    """
    Base class for cognitive agents in the collaboration system.
    """
    
    def __init__(self, agent_id: str, role: AgentRole, capabilities: List[str]):
        self.agent_id = agent_id
        self.role = role
        self.capabilities = capabilities
        self.synergy_engine = CognitiveSynergyEngine()
        
        # Communication
        self.message_queue = queue.PriorityQueue()
        self.sent_messages = deque(maxlen=1000)
        self.received_messages = deque(maxlen=1000)
        
        # Task management
        self.assigned_tasks: Dict[str, CollaborativeTask] = {}
        self.completed_tasks: List[str] = []
        self.performance_history = deque(maxlen=100)
        
        # Collaboration state
        self.known_agents: Dict[str, Dict[str, Any]] = {}
        self.trust_scores: Dict[str, float] = defaultdict(lambda: 0.5)
        self.collaboration_history: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        
        # Agent state
        self.is_active = True
        self.current_load = 0.0
        self.max_load = 1.0
        self.last_activity = datetime.now()
        
        # Start message processing
        self.message_processor = threading.Thread(target=self._process_messages)
        self.message_processor.daemon = True
        self.message_processor.start()
    
    def send_message(self, message: AgentMessage):
        """Send a message to another agent."""
        message.sender_id = self.agent_id
        message.timestamp = datetime.now()
        
        # Store in sent messages
        self.sent_messages.append(message)
        
        # In a real system, this would use a message broker
        # For demo, we'll use a simple callback mechanism
        if hasattr(self, 'message_router'):
            self.message_router(message)
    
    def receive_message(self, message: AgentMessage):
        """Receive a message from another agent."""
        priority = 1.0 - message.priority  # Higher priority = lower number for queue
        self.message_queue.put((priority, message.timestamp, message))
        self.received_messages.append(message)
    
    def _process_messages(self):
        """Process incoming messages."""
        while self.is_active:
            try:
                # Get message with timeout
                priority, timestamp, message = self.message_queue.get(timeout=1.0)
                self._handle_message(message)
                self.message_queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Agent {self.agent_id} message processing error: {e}")
    
    def _handle_message(self, message: AgentMessage):
        """Handle different types of messages."""
        self.last_activity = datetime.now()
        
        if message.message_type == MessageType.TASK_REQUEST:
            self._handle_task_request(message)
        elif message.message_type == MessageType.KNOWLEDGE_SHARE:
            self._handle_knowledge_share(message)
        elif message.message_type == MessageType.HELP_REQUEST:
            self._handle_help_request(message)
        elif message.message_type == MessageType.RESULT_SHARE:
            self._handle_result_share(message)
        elif message.message_type == MessageType.STATUS_UPDATE:
            self._handle_status_update(message)
        elif message.message_type == MessageType.COORDINATION:
            self._handle_coordination(message)
    
    def _handle_task_request(self, message: AgentMessage):
        """Handle task assignment requests."""
        task_data = message.content.get('task', {})
        
        # Evaluate if agent can handle the task
        can_handle = self._can_handle_task(task_data)
        
        response = AgentMessage(
            receiver_id=message.sender_id,
            message_type=MessageType.TASK_RESPONSE,
            content={
                'task_id': task_data.get('task_id'),
                'accepted': can_handle,
                'estimated_completion_time': self._estimate_completion_time(task_data),
                'current_load': self.current_load,
                'capabilities': self.capabilities
            }
        )
        
        self.send_message(response)
        
        if can_handle:
            # Accept the task
            task = CollaborativeTask(**task_data)
            self.assigned_tasks[task.task_id] = task
            self.current_load += self._calculate_task_load(task)
            
            # Start working on the task
            self._start_task_execution(task)
    
    def _handle_knowledge_share(self, message: AgentMessage):
        """Handle knowledge sharing from other agents."""
        knowledge = message.content.get('knowledge', {})
        source_agent = message.sender_id
        
        # Integrate knowledge into local memory
        memory = self.synergy_engine.get_memory()
        
        for fact, confidence in knowledge.items():
            # Create atom for shared knowledge
            atom = Atom(
                atom_type="SharedKnowledge",
                name=fact,
                truth_value=confidence,
                attention_value=0.3
            )
            atom_id = memory.add_atom(atom)
            
            # Link to source agent
            source_atom = Atom(
                atom_type="AgentNode",
                name=source_agent,
                truth_value=self.trust_scores[source_agent],
                attention_value=0.2
            )
            source_id = memory.add_atom(source_atom)
            memory.link_atoms(source_id, atom_id, "shared_knowledge")
        
        # Update trust score based on knowledge quality
        self._update_trust_score(source_agent, knowledge)
    
    def _handle_help_request(self, message: AgentMessage):
        """Handle requests for help from other agents."""
        help_type = message.content.get('help_type')
        problem_description = message.content.get('problem')
        
        # Evaluate if agent can provide help
        can_help = self._can_provide_help(help_type, problem_description)
        
        if can_help:
            # Generate help response
            help_content = self._generate_help(help_type, problem_description)
            
            response = AgentMessage(
                receiver_id=message.sender_id,
                message_type=MessageType.RESULT_SHARE,
                content={
                    'help_type': help_type,
                    'solution': help_content,
                    'confidence': 0.8
                }
            )
            
            self.send_message(response)
    
    def _handle_result_share(self, message: AgentMessage):
        """Handle shared results from other agents."""
        results = message.content.get('solution', {})
        confidence = message.content.get('confidence', 0.5)
        
        # Integrate results into current tasks if relevant
        for task_id, task in self.assigned_tasks.items():
            if self._is_result_relevant(results, task):
                task.results[f"shared_from_{message.sender_id}"] = {
                    'content': results,
                    'confidence': confidence,
                    'timestamp': datetime.now().isoformat()
                }
    
    def _handle_status_update(self, message: AgentMessage):
        """Handle status updates from other agents."""
        agent_id = message.sender_id
        status = message.content.get('status', {})
        
        # Update known agent information
        self.known_agents[agent_id] = {
            'last_seen': datetime.now(),
            'status': status,
            'capabilities': status.get('capabilities', []),
            'current_load': status.get('current_load', 0.0)
        }
    
    def _handle_coordination(self, message: AgentMessage):
        """Handle coordination messages."""
        coordination_type = message.content.get('type')
        
        if coordination_type == "task_decomposition":
            self._handle_task_decomposition(message)
        elif coordination_type == "resource_allocation":
            self._handle_resource_allocation(message)
        elif coordination_type == "consensus_building":
            self._handle_consensus_building(message)
    
    def _can_handle_task(self, task_data: Dict[str, Any]) -> bool:
        """Evaluate if agent can handle a given task."""
        requirements = task_data.get('requirements', [])
        
        # Check capability match
        capability_match = any(req in self.capabilities for req in requirements)
        
        # Check current load
        load_ok = self.current_load < self.max_load * 0.8
        
        # Check role appropriateness
        role_match = self._is_role_appropriate(task_data)
        
        return capability_match and load_ok and role_match
    
    def _is_role_appropriate(self, task_data: Dict[str, Any]) -> bool:
        """Check if agent's role is appropriate for the task."""
        task_type = task_data.get('type', 'general')
        
        role_task_mapping = {
            AgentRole.COORDINATOR: ['coordination', 'management', 'planning'],
            AgentRole.SPECIALIST: ['specialized', 'expert', 'technical'],
            AgentRole.GENERALIST: ['general', 'broad', 'diverse'],
            AgentRole.VALIDATOR: ['validation', 'verification', 'testing'],
            AgentRole.SYNTHESIZER: ['synthesis', 'integration', 'combination'],
            AgentRole.EXPLORER: ['exploration', 'discovery', 'research'],
            AgentRole.OPTIMIZER: ['optimization', 'improvement', 'efficiency']
        }
        
        appropriate_tasks = role_task_mapping.get(self.role, ['general'])
        return any(task_type in task for task in appropriate_tasks)
    
    def _estimate_completion_time(self, task_data: Dict[str, Any]) -> float:
        """Estimate time to complete a task."""
        base_time = 1.0  # Base time in hours
        
        # Adjust based on task complexity
        complexity = task_data.get('complexity', 0.5)
        base_time *= (1 + complexity)
        
        # Adjust based on current load
        base_time *= (1 + self.current_load)
        
        # Adjust based on capability match
        requirements = task_data.get('requirements', [])
        capability_score = sum(1 for req in requirements if req in self.capabilities) / max(len(requirements), 1)
        base_time *= (2 - capability_score)  # Better match = faster completion
        
        return base_time
    
    def _calculate_task_load(self, task: CollaborativeTask) -> float:
        """Calculate the load a task will place on the agent."""
        base_load = 0.2
        
        # Adjust based on task complexity
        complexity = len(task.requirements) / 10.0
        base_load += complexity * 0.3
        
        # Adjust based on subtasks
        base_load += len(task.subtasks) * 0.1
        
        return min(base_load, 0.8)  # Cap at 80% load per task
    
    def _start_task_execution(self, task: CollaborativeTask):
        """Start executing a task."""
        task.status = TaskStatus.IN_PROGRESS
        
        # Create execution thread
        execution_thread = threading.Thread(
            target=self._execute_task,
            args=(task,)
        )
        execution_thread.daemon = True
        execution_thread.start()
    
    def _execute_task(self, task: CollaborativeTask):
        """Execute a task (simplified implementation)."""
        try:
            # Simulate task execution
            execution_time = self._estimate_completion_time(task.__dict__)
            
            # Break down into steps
            steps = max(1, int(execution_time * 10))  # 10 steps per hour
            
            for step in range(steps):
                if not self.is_active:
                    break
                
                # Simulate work
                time.sleep(0.1)  # Reduced for demo
                
                # Check if help is needed
                if step > steps // 2 and np.random.random() < 0.1:  # 10% chance of needing help
                    self._request_help(task)
                
                # Share intermediate results occasionally
                if step % (steps // 3) == 0 and step > 0:
                    self._share_intermediate_results(task, step / steps)
            
            # Complete the task
            task.status = TaskStatus.COMPLETED
            task.completed_at = datetime.now()
            
            # Generate results
            task.results['final_result'] = self._generate_task_result(task)
            task.results['completion_time'] = execution_time
            task.results['agent_id'] = self.agent_id
            
            # Update performance history
            performance_score = self._calculate_performance_score(task)
            self.performance_history.append(performance_score)
            
            # Reduce current load
            self.current_load -= self._calculate_task_load(task)
            self.current_load = max(0.0, self.current_load)
            
            # Move to completed tasks
            self.completed_tasks.append(task.task_id)
            if task.task_id in self.assigned_tasks:
                del self.assigned_tasks[task.task_id]
            
            # Share results with other agents
            self._broadcast_task_completion(task)
            
        except Exception as e:
            logger.error(f"Task execution error for agent {self.agent_id}: {e}")
            task.status = TaskStatus.FAILED
    
    def _request_help(self, task: CollaborativeTask):
        """Request help from other agents."""
        help_message = AgentMessage(
            receiver_id="broadcast",  # Broadcast to all agents
            message_type=MessageType.HELP_REQUEST,
            content={
                'help_type': 'task_assistance',
                'problem': f"Need help with task: {task.description}",
                'task_id': task.task_id,
                'requirements': task.requirements
            },
            priority=0.7
        )
        
        self.send_message(help_message)
    
    def _share_intermediate_results(self, task: CollaborativeTask, progress: float):
        """Share intermediate results with other agents."""
        intermediate_results = {
            'task_id': task.task_id,
            'progress': progress,
            'partial_results': f"Intermediate result at {progress:.1%} completion",
            'insights': self._extract_insights(task, progress)
        }
        
        share_message = AgentMessage(
            receiver_id="broadcast",
            message_type=MessageType.RESULT_SHARE,
            content=intermediate_results,
            priority=0.4
        )
        
        self.send_message(share_message)
    
    def _generate_task_result(self, task: CollaborativeTask) -> Dict[str, Any]:
        """Generate final result for a completed task."""
        return {
            'task_description': task.description,
            'solution': f"Solution generated by {self.agent_id} using {self.role.value} approach",
            'quality_score': np.random.uniform(0.7, 0.95),  # Simulate quality
            'methodology': f"{self.role.value}_methodology",
            'confidence': np.random.uniform(0.8, 0.95),
            'resources_used': self.capabilities[:3]  # First 3 capabilities
        }
    
    def _calculate_performance_score(self, task: CollaborativeTask) -> float:
        """Calculate performance score for completed task."""
        base_score = 0.8
        
        # Adjust based on completion time vs estimate
        estimated_time = self._estimate_completion_time(task.__dict__)
        actual_time = (task.completed_at - task.created_at).total_seconds() / 3600.0
        
        if actual_time <= estimated_time:
            time_bonus = 0.2 * (1 - actual_time / estimated_time)
        else:
            time_penalty = 0.1 * (actual_time / estimated_time - 1)
            time_bonus = -min(time_penalty, 0.3)
        
        # Adjust based on result quality
        quality_score = task.results.get('final_result', {}).get('quality_score', 0.8)
        quality_bonus = (quality_score - 0.8) * 0.5
        
        final_score = base_score + time_bonus + quality_bonus
        return max(0.0, min(1.0, final_score))
    
    def _broadcast_task_completion(self, task: CollaborativeTask):
        """Broadcast task completion to other agents."""
        completion_message = AgentMessage(
            receiver_id="broadcast",
            message_type=MessageType.RESULT_SHARE,
            content={
                'task_id': task.task_id,
                'status': 'completed',
                'results': task.results,
                'lessons_learned': self._extract_lessons_learned(task)
            },
            priority=0.6
        )
        
        self.send_message(completion_message)
    
    def _extract_insights(self, task: CollaborativeTask, progress: float) -> List[str]:
        """Extract insights from task execution."""
        insights = [
            f"Task complexity appears to be {['low', 'medium', 'high'][int(progress * 3)]}",
            f"Agent role {self.role.value} is {'well' if progress > 0.5 else 'poorly'} suited for this task",
            f"Collaboration {'beneficial' if len(task.results) > 1 else 'not yet utilized'}"
        ]
        return insights
    
    def _extract_lessons_learned(self, task: CollaborativeTask) -> List[str]:
        """Extract lessons learned from completed task."""
        lessons = [
            f"Task type '{task.description[:20]}...' requires {len(task.requirements)} key capabilities",
            f"Completion time was {'faster' if len(self.performance_history) > 0 and self.performance_history[-1] > 0.8 else 'slower'} than expected",
            f"Collaboration {'improved' if len(task.results) > 2 else 'had minimal impact on'} task outcome"
        ]
        return lessons
    
    def _can_provide_help(self, help_type: str, problem: str) -> bool:
        """Determine if agent can provide help for a specific problem."""
        # Check if agent has relevant capabilities
        problem_keywords = problem.lower().split()
        capability_match = any(cap.lower() in problem_keywords for cap in self.capabilities)
        
        # Check current load
        load_ok = self.current_load < self.max_load * 0.9
        
        # Check role appropriateness for help type
        help_roles = {
            'task_assistance': [AgentRole.GENERALIST, AgentRole.SPECIALIST],
            'validation': [AgentRole.VALIDATOR],
            'optimization': [AgentRole.OPTIMIZER],
            'coordination': [AgentRole.COORDINATOR]
        }
        
        role_match = self.role in help_roles.get(help_type, [AgentRole.GENERALIST])
        
        return capability_match and load_ok and role_match
    
    def _generate_help(self, help_type: str, problem: str) -> Dict[str, Any]:
        """Generate help content for a specific problem."""
        return {
            'advice': f"Based on my {self.role.value} expertise, I suggest focusing on {self.capabilities[0]}",
            'resources': self.capabilities[:2],
            'alternative_approaches': [f"Approach using {cap}" for cap in self.capabilities[:3]],
            'confidence': 0.8
        }
    
    def _is_result_relevant(self, results: Dict[str, Any], task: CollaborativeTask) -> bool:
        """Check if shared results are relevant to current task."""
        # Simple keyword matching
        result_text = str(results).lower()
        task_text = task.description.lower()
        
        common_words = set(result_text.split()) & set(task_text.split())
        return len(common_words) > 2
    
    def _update_trust_score(self, agent_id: str, knowledge: Dict[str, Any]):
        """Update trust score for another agent based on knowledge quality."""
        # Simple trust update based on knowledge consistency
        current_trust = self.trust_scores[agent_id]
        
        # Simulate knowledge quality assessment
        quality_score = np.random.uniform(0.6, 0.9)  # In practice, this would be more sophisticated
        
        # Update trust with exponential moving average
        alpha = 0.1
        new_trust = alpha * quality_score + (1 - alpha) * current_trust
        self.trust_scores[agent_id] = new_trust
    
    def get_status(self) -> Dict[str, Any]:
        """Get current agent status."""
        return {
            'agent_id': self.agent_id,
            'role': self.role.value,
            'capabilities': self.capabilities,
            'current_load': self.current_load,
            'active_tasks': len(self.assigned_tasks),
            'completed_tasks': len(self.completed_tasks),
            'average_performance': statistics.mean(self.performance_history) if self.performance_history else 0.5,
            'last_activity': self.last_activity.isoformat(),
            'known_agents': len(self.known_agents)
        }
    
    def shutdown(self):
        """Shutdown the agent."""
        self.is_active = False


class MultiAgentCoordinator:
    """
    Coordinates multiple cognitive agents for collaborative problem-solving.
    """
    
    def __init__(self):
        self.agents: Dict[str, CognitiveAgent] = {}
        self.tasks: Dict[str, CollaborativeTask] = {}
        self.message_router = self._route_message
        self.collaboration_history = []
        self.performance_metrics = defaultdict(list)
        
    def register_agent(self, agent: CognitiveAgent):
        """Register an agent with the coordinator."""
        self.agents[agent.agent_id] = agent
        agent.message_router = self.message_router
        
        logger.info(f"Registered agent {agent.agent_id} with role {agent.role.value}")
    
    def _route_message(self, message: AgentMessage):
        """Route messages between agents."""
        if message.receiver_id == "broadcast":
            # Broadcast to all agents except sender
            for agent_id, agent in self.agents.items():
                if agent_id != message.sender_id:
                    agent.receive_message(message)
        elif message.receiver_id in self.agents:
            # Direct message
            self.agents[message.receiver_id].receive_message(message)
        else:
            logger.warning(f"Message to unknown agent: {message.receiver_id}")
    
    def create_collaborative_task(self, 
                                description: str,
                                requirements: List[str],
                                priority: float = 0.5,
                                deadline: Optional[datetime] = None) -> str:
        """Create a new collaborative task."""
        task = CollaborativeTask(
            description=description,
            requirements=requirements,
            priority=priority,
            deadline=deadline
        )
        
        self.tasks[task.task_id] = task
        
        # Decompose task if complex
        if len(requirements) > 3:
            subtasks = self._decompose_task(task)
            for subtask in subtasks:
                task.add_subtask(subtask)
        
        # Assign task to appropriate agents
        self._assign_task(task)
        
        return task.task_id
    
    def _decompose_task(self, task: CollaborativeTask) -> List[CollaborativeTask]:
        """Decompose a complex task into subtasks."""
        subtasks = []
        
        # Group requirements into logical subtasks
        requirement_groups = self._group_requirements(task.requirements)
        
        for i, req_group in enumerate(requirement_groups):
            subtask = CollaborativeTask(
                description=f"Subtask {i+1} of {task.description}",
                requirements=req_group,
                priority=task.priority,
                deadline=task.deadline
            )
            subtasks.append(subtask)
        
        return subtasks
    
    def _group_requirements(self, requirements: List[str]) -> List[List[str]]:
        """Group related requirements together."""
        # Simple grouping by similarity (in practice, this would be more sophisticated)
        groups = []
        used = set()
        
        for req in requirements:
            if req in used:
                continue
            
            group = [req]
            used.add(req)
            
            # Find similar requirements
            for other_req in requirements:
                if other_req not in used and self._requirements_similar(req, other_req):
                    group.append(other_req)
                    used.add(other_req)
            
            groups.append(group)
        
        return groups
    
    def _requirements_similar(self, req1: str, req2: str) -> bool:
        """Check if two requirements are similar."""
        # Simple word overlap check
        words1 = set(req1.lower().split())
        words2 = set(req2.lower().split())
        overlap = len(words1 & words2)
        return overlap > 0
    
    def _assign_task(self, task: CollaborativeTask):
        """Assign task to appropriate agents."""
        # Find suitable agents
        suitable_agents = self._find_suitable_agents(task)
        
        if not suitable_agents:
            logger.warning(f"No suitable agents found for task {task.task_id}")
            return
        
        # Send task requests
        for agent_id in suitable_agents[:3]:  # Limit to top 3 agents
            task_request = AgentMessage(
                receiver_id=agent_id,
                message_type=MessageType.TASK_REQUEST,
                content={'task': task.__dict__},
                priority=task.priority,
                requires_response=True
            )
            
            self.agents[agent_id].receive_message(task_request)
    
    def _find_suitable_agents(self, task: CollaborativeTask) -> List[str]:
        """Find agents suitable for a task."""
        suitable_agents = []
        
        for agent_id, agent in self.agents.items():
            # Check capability match
            capability_score = self._calculate_capability_match(agent, task)
            
            # Check availability
            availability_score = 1.0 - agent.current_load
            
            # Check performance history
            performance_score = statistics.mean(agent.performance_history) if agent.performance_history else 0.5
            
            # Combined suitability score
            suitability = (capability_score * 0.5 + 
                         availability_score * 0.3 + 
                         performance_score * 0.2)
            
            if suitability > 0.4:  # Threshold for suitability
                suitable_agents.append((agent_id, suitability))
        
        # Sort by suitability and return agent IDs
        suitable_agents.sort(key=lambda x: x[1], reverse=True)
        return [agent_id for agent_id, _ in suitable_agents]
    
    def _calculate_capability_match(self, agent: CognitiveAgent, task: CollaborativeTask) -> float:
        """Calculate how well an agent's capabilities match task requirements."""
        if not task.requirements:
            return 0.5
        
        matches = sum(1 for req in task.requirements if req in agent.capabilities)
        return matches / len(task.requirements)
    
    def get_collaboration_status(self) -> Dict[str, Any]:
        """Get overall collaboration status."""
        active_tasks = [task for task in self.tasks.values() if task.status == TaskStatus.IN_PROGRESS]
        completed_tasks = [task for task in self.tasks.values() if task.status == TaskStatus.COMPLETED]
        
        agent_loads = [agent.current_load for agent in self.agents.values()]
        agent_performances = []
        for agent in self.agents.values():
            if agent.performance_history:
                agent_performances.append(statistics.mean(agent.performance_history))
        
        return {
            'total_agents': len(self.agents),
            'active_tasks': len(active_tasks),
            'completed_tasks': len(completed_tasks),
            'average_agent_load': statistics.mean(agent_loads) if agent_loads else 0.0,
            'average_performance': statistics.mean(agent_performances) if agent_performances else 0.5,
            'collaboration_efficiency': self._calculate_collaboration_efficiency()
        }
    
    def _calculate_collaboration_efficiency(self) -> float:
        """Calculate overall collaboration efficiency."""
        if not self.tasks:
            return 0.5
        
        completed_tasks = [task for task in self.tasks.values() if task.status == TaskStatus.COMPLETED]
        
        if not completed_tasks:
            return 0.3
        
        # Calculate efficiency based on task completion rate and quality
        completion_rate = len(completed_tasks) / len(self.tasks)
        
        quality_scores = []
        for task in completed_tasks:
            result = task.results.get('final_result', {})
            quality = result.get('quality_score', 0.8)
            quality_scores.append(quality)
        
        average_quality = statistics.mean(quality_scores) if quality_scores else 0.8
        
        efficiency = (completion_rate * 0.6 + average_quality * 0.4)
        return efficiency
    
    def shutdown_all_agents(self):
        """Shutdown all agents."""
        for agent in self.agents.values():
            agent.shutdown()


def demonstrate_multi_agent_collaboration():
    """
    Demonstrate multi-agent cognitive collaboration.
    """
    print("=== Multi-Agent Cognitive Collaboration Demonstration ===\n")
    
    # Create coordinator
    coordinator = MultiAgentCoordinator()
    
    # Create diverse agents
    agents = [
        CognitiveAgent("coordinator_001", AgentRole.COORDINATOR, 
                      ["planning", "coordination", "resource_allocation"]),
        CognitiveAgent("specialist_001", AgentRole.SPECIALIST, 
                      ["machine_learning", "data_analysis", "pattern_recognition"]),
        CognitiveAgent("specialist_002", AgentRole.SPECIALIST, 
                      ["symbolic_reasoning", "logic", "knowledge_representation"]),
        CognitiveAgent("generalist_001", AgentRole.GENERALIST, 
                      ["problem_solving", "integration", "communication"]),
        CognitiveAgent("validator_001", AgentRole.VALIDATOR, 
                      ["validation", "testing", "quality_assurance"]),
        CognitiveAgent("synthesizer_001", AgentRole.SYNTHESIZER, 
                      ["synthesis", "integration", "combination"])
    ]
    
    # Register agents
    for agent in agents:
        coordinator.register_agent(agent)
    
    print(f"Created and registered {len(agents)} agents:")
    for agent in agents:
        print(f"  {agent.agent_id}: {agent.role.value} - {agent.capabilities}")
    
    # Create collaborative tasks
    print("\nCreating collaborative tasks...")
    
    task1_id = coordinator.create_collaborative_task(
        description="Develop a hybrid AI system combining neural networks and symbolic reasoning",
        requirements=["machine_learning", "symbolic_reasoning", "integration", "validation"],
        priority=0.8
    )
    
    task2_id = coordinator.create_collaborative_task(
        description="Analyze complex dataset and extract meaningful patterns",
        requirements=["data_analysis", "pattern_recognition", "validation"],
        priority=0.6
    )
    
    task3_id = coordinator.create_collaborative_task(
        description="Design cognitive architecture for multi-modal reasoning",
        requirements=["planning", "knowledge_representation", "synthesis", "coordination"],
        priority=0.9
    )
    
    print(f"Created tasks: {task1_id[:8]}..., {task2_id[:8]}..., {task3_id[:8]}...")
    
    # Monitor collaboration for a period
    print("\nMonitoring collaboration progress...")
    
    for iteration in range(10):
        time.sleep(0.5)  # Reduced for demo
        
        status = coordinator.get_collaboration_status()
        print(f"Iteration {iteration + 1}:")
        print(f"  Active tasks: {status['active_tasks']}")
        print(f"  Completed tasks: {status['completed_tasks']}")
        print(f"  Average agent load: {status['average_agent_load']:.2f}")
        print(f"  Average performance: {status['average_performance']:.2f}")
        print(f"  Collaboration efficiency: {status['collaboration_efficiency']:.2f}")
        
        # Show agent statuses
        if iteration % 3 == 0:
            print("  Agent statuses:")
            for agent in agents[:3]:  # Show first 3 agents
                agent_status = agent.get_status()
                print(f"    {agent_status['agent_id']}: Load {agent_status['current_load']:.2f}, "
                      f"Tasks {agent_status['active_tasks']}, "
                      f"Performance {agent_status['average_performance']:.2f}")
    
    # Final status
    print("\nFinal collaboration results:")
    final_status = coordinator.get_collaboration_status()
    
    for key, value in final_status.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.3f}")
        else:
            print(f"  {key}: {value}")
    
    # Show completed task results
    completed_tasks = [task for task in coordinator.tasks.values() if task.status == TaskStatus.COMPLETED]
    print(f"\nCompleted {len(completed_tasks)} tasks:")
    
    for task in completed_tasks:
        print(f"  Task: {task.description[:50]}...")
        result = task.results.get('final_result', {})
        print(f"    Quality: {result.get('quality_score', 0.0):.2f}")
        print(f"    Confidence: {result.get('confidence', 0.0):.2f}")
        print(f"    Completed by: {result.get('agent_id', 'unknown')}")
    
    # Shutdown
    print("\nShutting down agents...")
    coordinator.shutdown_all_agents()
    
    print("\n=== Multi-Agent Collaboration Demo Complete ===")


if __name__ == "__main__":
    demonstrate_multi_agent_collaboration()

"""
Cognitive Component Discovery and Registration System
======================================================

Automatically discovers, validates, and registers cognitive components
in the OpenCog Collection for dynamic orchestration and synergy.

This system enables plug-and-play cognitive extensions by:
- Scanning Python modules for cognitive components
- Validating component interfaces and capabilities
- Building a dynamic registry for orchestration
- Detecting version compatibility

Author: OpenCog Collection Contributors
License: GPL-3.0+
"""

import os
import sys
import importlib.util
import inspect
import json
from typing import Dict, List, Any, Optional, Type, Callable
from dataclasses import dataclass, field, asdict
from pathlib import Path
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ComponentCapabilities:
    """Describes the capabilities of a cognitive component."""
    has_agent_interface: bool = False
    has_arena_interface: bool = False
    has_relation_interface: bool = False
    supports_hypergraph: bool = False
    supports_neural_symbolic: bool = False
    supports_multi_agent: bool = False
    supports_mcp: bool = False
    supports_persistence: bool = False
    supports_monitoring: bool = False
    custom_capabilities: List[str] = field(default_factory=list)


@dataclass
class ComponentMetadata:
    """Metadata for a discovered cognitive component."""
    name: str
    module_path: str
    class_name: str
    version: str = "0.1.0"
    description: str = ""
    capabilities: ComponentCapabilities = field(default_factory=ComponentCapabilities)
    dependencies: List[str] = field(default_factory=list)
    interfaces: List[str] = field(default_factory=list)
    discovered_at: str = field(default_factory=lambda: datetime.now().isoformat())


class CognitiveComponentRegistry:
    """
    Central registry for cognitive components in the OCC ecosystem.
    Enables dynamic discovery, validation, and orchestration.
    """
    
    def __init__(self, registry_path: str = "cognitive_registry.json"):
        self.registry_path = registry_path
        self.components: Dict[str, ComponentMetadata] = {}
        self.load_registry()
    
    def load_registry(self):
        """Load existing registry from disk."""
        if os.path.exists(self.registry_path):
            try:
                with open(self.registry_path, 'r') as f:
                    data = json.load(f)
                    for name, comp_data in data.items():
                        # Reconstruct ComponentCapabilities
                        caps_data = comp_data.pop('capabilities', {})
                        capabilities = ComponentCapabilities(**caps_data)
                        comp_data['capabilities'] = capabilities
                        self.components[name] = ComponentMetadata(**comp_data)
                logger.info(f"Loaded {len(self.components)} components from registry")
            except Exception as e:
                logger.error(f"Failed to load registry: {e}")
    
    def save_registry(self):
        """Save registry to disk."""
        try:
            data = {}
            for name, comp in self.components.items():
                comp_dict = asdict(comp)
                data[name] = comp_dict
            
            with open(self.registry_path, 'w') as f:
                json.dump(data, f, indent=2)
            logger.info(f"Saved {len(self.components)} components to registry")
        except Exception as e:
            logger.error(f"Failed to save registry: {e}")
    
    def register_component(self, metadata: ComponentMetadata):
        """Register a cognitive component."""
        self.components[metadata.name] = metadata
        logger.info(f"Registered component: {metadata.name}")
    
    def get_component(self, name: str) -> Optional[ComponentMetadata]:
        """Retrieve component metadata by name."""
        return self.components.get(name)
    
    def list_components(self) -> List[str]:
        """List all registered component names."""
        return list(self.components.keys())
    
    def find_by_capability(self, capability: str) -> List[ComponentMetadata]:
        """Find components with a specific capability."""
        results = []
        for comp in self.components.values():
            caps = comp.capabilities
            if hasattr(caps, capability) and getattr(caps, capability):
                results.append(comp)
            elif capability in caps.custom_capabilities:
                results.append(comp)
        return results


class CognitiveComponentDiscovery:
    """
    Discovers cognitive components by scanning Python modules
    and analyzing their structure and interfaces.
    """
    
    # Known cognitive component indicators
    COGNITIVE_INDICATORS = [
        'cognitive', 'synergy', 'agent', 'arena', 'relation',
        'hypergraph', 'neural', 'symbolic', 'meta', 'awareness',
        'orchestrator', 'monitoring', 'attention', 'mcp'
    ]
    
    # Required interfaces for different component types
    INTERFACE_PATTERNS = {
        'agent': ['process', 'update', 'action'],
        'arena': ['state', 'memory', 'knowledge'],
        'relation': ['coherence', 'integrate', 'emerge'],
        'hypergraph': ['add_node', 'add_edge', 'query'],
        'monitor': ['collect_metrics', 'report'],
    }
    
    def __init__(self, registry: CognitiveComponentRegistry):
        self.registry = registry
        self.discovered_modules: List[str] = []
    
    def scan_directory(self, directory: str = ".") -> int:
        """
        Scan a directory for cognitive components.
        Returns the number of components discovered.
        """
        discovered_count = 0
        
        for py_file in Path(directory).glob("*.py"):
            if py_file.name.startswith("_"):
                continue
            
            # Check if filename suggests cognitive component
            if any(indicator in py_file.stem.lower() 
                   for indicator in self.COGNITIVE_INDICATORS):
                
                try:
                    metadata = self.analyze_module(py_file)
                    if metadata:
                        self.registry.register_component(metadata)
                        discovered_count += 1
                except Exception as e:
                    logger.warning(f"Failed to analyze {py_file}: {e}")
        
        return discovered_count
    
    def analyze_module(self, module_path: Path) -> Optional[ComponentMetadata]:
        """
        Analyze a Python module to extract cognitive component metadata.
        """
        module_name = module_path.stem
        
        try:
            # Import the module
            spec = importlib.util.spec_from_file_location(module_name, module_path)
            if not spec or not spec.loader:
                return None
            
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            # Extract module docstring
            description = inspect.getdoc(module) or ""
            
            # Analyze classes in the module
            capabilities = self.detect_capabilities(module)
            interfaces = self.detect_interfaces(module)
            dependencies = self.detect_dependencies(module_path)
            
            # Find main class (heuristic: largest class or class matching module name)
            main_class = self.find_main_class(module, module_name)
            
            metadata = ComponentMetadata(
                name=module_name,
                module_path=str(module_path),
                class_name=main_class.__name__ if main_class else "",
                description=description[:200],  # First 200 chars
                capabilities=capabilities,
                interfaces=interfaces,
                dependencies=dependencies
            )
            
            return metadata
            
        except Exception as e:
            logger.error(f"Error analyzing module {module_path}: {e}")
            return None
    
    def detect_capabilities(self, module) -> ComponentCapabilities:
        """Detect capabilities by analyzing module structure."""
        capabilities = ComponentCapabilities()
        
        # Get all classes and functions in module
        members = inspect.getmembers(module)
        
        for name, obj in members:
            if inspect.isclass(obj) or inspect.isfunction(obj):
                name_lower = name.lower()
                
                # Check for AAR components
                if 'agent' in name_lower:
                    capabilities.has_agent_interface = True
                if 'arena' in name_lower:
                    capabilities.has_arena_interface = True
                if 'relation' in name_lower:
                    capabilities.has_relation_interface = True
                
                # Check for specific capabilities
                if 'hypergraph' in name_lower:
                    capabilities.supports_hypergraph = True
                if 'neural' in name_lower or 'symbolic' in name_lower:
                    capabilities.supports_neural_symbolic = True
                if 'multi' in name_lower and 'agent' in name_lower:
                    capabilities.supports_multi_agent = True
                if 'mcp' in name_lower:
                    capabilities.supports_mcp = True
                if 'persist' in name_lower or 'storage' in name_lower:
                    capabilities.supports_persistence = True
                if 'monitor' in name_lower or 'metric' in name_lower:
                    capabilities.supports_monitoring = True
        
        return capabilities
    
    def detect_interfaces(self, module) -> List[str]:
        """Detect implemented interfaces by analyzing method signatures."""
        interfaces = []
        
        for name, obj in inspect.getmembers(module, inspect.isclass):
            methods = [m for m, _ in inspect.getmembers(obj, inspect.ismethod)]
            methods.extend([m for m, _ in inspect.getmembers(obj, inspect.isfunction)])
            
            # Check against known interface patterns
            for interface_name, required_methods in self.INTERFACE_PATTERNS.items():
                if all(any(req in m.lower() for m in methods) 
                       for req in required_methods):
                    interfaces.append(interface_name)
        
        return list(set(interfaces))
    
    def detect_dependencies(self, module_path: Path) -> List[str]:
        """Detect dependencies by parsing import statements."""
        dependencies = []
        
        try:
            with open(module_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line.startswith('import ') or line.startswith('from '):
                        # Extract module name
                        parts = line.split()
                        if len(parts) >= 2:
                            dep = parts[1].split('.')[0]
                            if dep not in ['os', 'sys', 'typing', 'dataclasses']:
                                dependencies.append(dep)
        except Exception as e:
            logger.warning(f"Failed to parse dependencies from {module_path}: {e}")
        
        return list(set(dependencies))
    
    def find_main_class(self, module, module_name: str) -> Optional[Type]:
        """Find the main class in a module (heuristic-based)."""
        classes = [obj for name, obj in inspect.getmembers(module, inspect.isclass)
                   if obj.__module__ == module.__name__]
        
        if not classes:
            return None
        
        # Prefer class with name matching module
        for cls in classes:
            if cls.__name__.lower() == module_name.lower().replace('_', ''):
                return cls
        
        # Otherwise return largest class (most methods)
        return max(classes, key=lambda c: len(inspect.getmembers(c, inspect.ismethod)))


def main():
    """Main entry point for cognitive component discovery."""
    print("=" * 60)
    print("OpenCog Collection - Cognitive Component Discovery")
    print("=" * 60)
    print()
    
    # Initialize registry
    registry = CognitiveComponentRegistry()
    
    # Initialize discovery
    discovery = CognitiveComponentDiscovery(registry)
    
    # Scan current directory
    print("Scanning for cognitive components...")
    count = discovery.scan_directory(".")
    
    print(f"\n✓ Discovered {count} cognitive components")
    
    # Save registry
    registry.save_registry()
    print(f"✓ Registry saved to {registry.registry_path}")
    
    # Print summary
    print("\n" + "=" * 60)
    print("Component Summary by Capability")
    print("=" * 60)
    
    capability_counts = {
        'Agent Interface': len(registry.find_by_capability('has_agent_interface')),
        'Arena Interface': len(registry.find_by_capability('has_arena_interface')),
        'Relation Interface': len(registry.find_by_capability('has_relation_interface')),
        'Hypergraph Support': len(registry.find_by_capability('supports_hypergraph')),
        'Neural-Symbolic': len(registry.find_by_capability('supports_neural_symbolic')),
        'Multi-Agent': len(registry.find_by_capability('supports_multi_agent')),
        'MCP Integration': len(registry.find_by_capability('supports_mcp')),
        'Persistence': len(registry.find_by_capability('supports_persistence')),
        'Monitoring': len(registry.find_by_capability('supports_monitoring')),
    }
    
    for capability, count in capability_counts.items():
        print(f"  {capability:.<40} {count:>3}")
    
    print("\n" + "=" * 60)
    print("Registered Components:")
    print("=" * 60)
    
    for name in sorted(registry.list_components()):
        comp = registry.get_component(name)
        if comp:
            print(f"  • {name}")
            if comp.description:
                desc = comp.description.split('\n')[0][:60]
                print(f"    {desc}...")
    
    print("\n✓ Discovery complete!")


if __name__ == "__main__":
    main()

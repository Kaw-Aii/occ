"""
MCP Server Configuration and Authentication
===========================================

Enhanced configuration for the AtomSpace MCP server with:
- API key authentication
- Rate limiting
- Request validation
- Logging and monitoring
- Error handling

Author: OpenCog Collection Contributors
License: GPL-3.0+
"""

import os
import hashlib
import secrets
import time
from typing import Dict, List, Optional, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict, deque
import logging
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class APIKey:
    """Represents an API key for MCP server access."""
    key_id: str
    key_hash: str
    name: str
    created_at: datetime = field(default_factory=datetime.now)
    last_used: Optional[datetime] = None
    rate_limit: int = 100  # requests per minute
    permissions: Set[str] = field(default_factory=set)
    active: bool = True


@dataclass
class RateLimitState:
    """Tracks rate limit state for an API key."""
    request_count: int = 0
    window_start: datetime = field(default_factory=datetime.now)
    blocked_until: Optional[datetime] = None


class MCPServerConfig:
    """Configuration manager for MCP server."""
    
    def __init__(self, config_file: Optional[str] = None):
        self.config_file = config_file or os.getenv(
            'MCP_CONFIG_FILE',
            '/etc/opencog/mcp_server.json'
        )
        
        # Default configuration
        self.config = {
            'server': {
                'host': '0.0.0.0',
                'port': 8765,
                'max_connections': 100,
                'timeout': 30
            },
            'security': {
                'require_auth': True,
                'api_key_header': 'X-API-Key',
                'rate_limit_enabled': True,
                'rate_limit_window': 60,  # seconds
                'max_request_size': 1048576  # 1MB
            },
            'database': {
                'supabase_url': os.getenv('SUPABASE_URL', ''),
                'supabase_key': os.getenv('SUPABASE_KEY', ''),
                'connection_pool_size': 10
            },
            'logging': {
                'level': 'INFO',
                'file': '/var/log/opencog/mcp_server.log',
                'max_size': 10485760,  # 10MB
                'backup_count': 5
            },
            'features': {
                'enable_caching': True,
                'cache_ttl': 300,  # seconds
                'enable_metrics': True,
                'metrics_interval': 60  # seconds
            }
        }
        
        # Load config from file if exists
        self.load_config()
    
    def load_config(self):
        """Load configuration from file."""
        try:
            if os.path.exists(self.config_file):
                with open(self.config_file, 'r') as f:
                    loaded_config = json.load(f)
                    self.config.update(loaded_config)
                logger.info(f"Configuration loaded from {self.config_file}")
        except Exception as e:
            logger.warning(f"Could not load config file: {e}, using defaults")
    
    def save_config(self):
        """Save configuration to file."""
        try:
            os.makedirs(os.path.dirname(self.config_file), exist_ok=True)
            with open(self.config_file, 'w') as f:
                json.dump(self.config, f, indent=2)
            logger.info(f"Configuration saved to {self.config_file}")
        except Exception as e:
            logger.error(f"Could not save config file: {e}")
    
    def get(self, key_path: str, default=None):
        """Get configuration value by dot-separated path."""
        keys = key_path.split('.')
        value = self.config
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        return value
    
    def set(self, key_path: str, value):
        """Set configuration value by dot-separated path."""
        keys = key_path.split('.')
        config = self.config
        for key in keys[:-1]:
            if key not in config:
                config[key] = {}
            config = config[key]
        config[keys[-1]] = value


class MCPAuthManager:
    """Manages authentication for MCP server."""
    
    def __init__(self):
        self.api_keys: Dict[str, APIKey] = {}
        self.rate_limits: Dict[str, RateLimitState] = defaultdict(RateLimitState)
        self.access_log: deque = deque(maxlen=1000)
        
        # Load API keys from environment or file
        self._load_api_keys()
    
    def _load_api_keys(self):
        """Load API keys from configuration."""
        # Check for environment variable
        env_key = os.getenv('MCP_API_KEY')
        if env_key:
            self.add_api_key(
                name='default',
                key=env_key,
                permissions={'query_atoms', 'create_atom', 'link_atoms'}
            )
        
        # Load from file if exists
        keys_file = os.getenv('MCP_KEYS_FILE', '/etc/opencog/mcp_keys.json')
        if os.path.exists(keys_file):
            try:
                with open(keys_file, 'r') as f:
                    keys_data = json.load(f)
                    for key_data in keys_data.get('keys', []):
                        self.api_keys[key_data['key_id']] = APIKey(
                            key_id=key_data['key_id'],
                            key_hash=key_data['key_hash'],
                            name=key_data['name'],
                            rate_limit=key_data.get('rate_limit', 100),
                            permissions=set(key_data.get('permissions', [])),
                            active=key_data.get('active', True)
                        )
                logger.info(f"Loaded {len(self.api_keys)} API keys from {keys_file}")
            except Exception as e:
                logger.error(f"Could not load API keys: {e}")
    
    def generate_api_key(self) -> str:
        """Generate a new API key."""
        return f"occ_{secrets.token_urlsafe(32)}"
    
    def hash_key(self, key: str) -> str:
        """Hash an API key for storage."""
        return hashlib.sha256(key.encode()).hexdigest()
    
    def add_api_key(self, name: str, key: Optional[str] = None,
                   permissions: Optional[Set[str]] = None,
                   rate_limit: int = 100) -> str:
        """Add a new API key."""
        if key is None:
            key = self.generate_api_key()
        
        key_id = f"key_{len(self.api_keys) + 1}"
        key_hash = self.hash_key(key)
        
        api_key = APIKey(
            key_id=key_id,
            key_hash=key_hash,
            name=name,
            rate_limit=rate_limit,
            permissions=permissions or set(),
            active=True
        )
        
        self.api_keys[key_id] = api_key
        logger.info(f"API key added: {key_id} ({name})")
        
        return key
    
    def validate_key(self, key: str) -> Optional[APIKey]:
        """Validate an API key."""
        key_hash = self.hash_key(key)
        
        for api_key in self.api_keys.values():
            if api_key.key_hash == key_hash and api_key.active:
                api_key.last_used = datetime.now()
                return api_key
        
        return None
    
    def check_rate_limit(self, key_id: str, rate_limit: int) -> bool:
        """Check if request is within rate limit."""
        state = self.rate_limits[key_id]
        now = datetime.now()
        
        # Check if blocked
        if state.blocked_until and now < state.blocked_until:
            return False
        
        # Reset window if needed
        if (now - state.window_start).total_seconds() >= 60:
            state.request_count = 0
            state.window_start = now
            state.blocked_until = None
        
        # Check limit
        if state.request_count >= rate_limit:
            state.blocked_until = state.window_start + timedelta(seconds=60)
            logger.warning(f"Rate limit exceeded for {key_id}")
            return False
        
        state.request_count += 1
        return True
    
    def check_permission(self, api_key: APIKey, operation: str) -> bool:
        """Check if API key has permission for operation."""
        if not api_key.permissions:
            return True  # No restrictions
        return operation in api_key.permissions
    
    def log_access(self, key_id: str, operation: str, success: bool,
                  details: Optional[str] = None):
        """Log an access attempt."""
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'key_id': key_id,
            'operation': operation,
            'success': success,
            'details': details
        }
        self.access_log.append(log_entry)
    
    def get_access_log(self, limit: int = 100) -> List[Dict]:
        """Get recent access log entries."""
        return list(self.access_log)[-limit:]


class MCPRequestValidator:
    """Validates MCP requests."""
    
    def __init__(self, max_request_size: int = 1048576):
        self.max_request_size = max_request_size
    
    def validate_request(self, request_data: Dict) -> Tuple[bool, Optional[str]]:
        """Validate a request."""
        # Check size
        request_size = len(json.dumps(request_data))
        if request_size > self.max_request_size:
            return False, f"Request too large: {request_size} bytes"
        
        # Check required fields
        if 'tool' not in request_data:
            return False, "Missing 'tool' field"
        
        if 'arguments' not in request_data:
            return False, "Missing 'arguments' field"
        
        # Validate tool name
        tool = request_data['tool']
        if not isinstance(tool, str) or not tool:
            return False, "Invalid tool name"
        
        # Validate arguments
        arguments = request_data['arguments']
        if not isinstance(arguments, dict):
            return False, "Arguments must be a dictionary"
        
        return True, None
    
    def sanitize_input(self, value: Any) -> Any:
        """Sanitize input values."""
        if isinstance(value, str):
            # Remove potentially dangerous characters
            return value.replace('\x00', '').strip()
        elif isinstance(value, dict):
            return {k: self.sanitize_input(v) for k, v in value.items()}
        elif isinstance(value, list):
            return [self.sanitize_input(v) for v in value]
        else:
            return value


# Example usage
if __name__ == "__main__":
    print("=== MCP Server Configuration ===\n")
    
    # Initialize configuration
    config = MCPServerConfig()
    print(f"Server host: {config.get('server.host')}")
    print(f"Server port: {config.get('server.port')}")
    print(f"Auth required: {config.get('security.require_auth')}")
    print()
    
    # Initialize authentication
    auth = MCPAuthManager()
    
    # Generate and add API key
    new_key = auth.add_api_key(
        name='test_client',
        permissions={'query_atoms', 'create_atom'},
        rate_limit=50
    )
    print(f"Generated API key: {new_key}")
    print()
    
    # Validate key
    api_key = auth.validate_key(new_key)
    if api_key:
        print(f"Key validated: {api_key.name}")
        print(f"Permissions: {api_key.permissions}")
        print(f"Rate limit: {api_key.rate_limit} req/min")
    print()
    
    # Check rate limit
    for i in range(5):
        allowed = auth.check_rate_limit(api_key.key_id, api_key.rate_limit)
        print(f"Request {i+1}: {'Allowed' if allowed else 'Blocked'}")
    print()
    
    # Check permission
    has_perm = auth.check_permission(api_key, 'query_atoms')
    print(f"Has 'query_atoms' permission: {has_perm}")
    
    has_perm = auth.check_permission(api_key, 'delete_atoms')
    print(f"Has 'delete_atoms' permission: {has_perm}")
    print()
    
    # Validate request
    validator = MCPRequestValidator()
    test_request = {
        'tool': 'query_atoms',
        'arguments': {'atom_type': 'ConceptNode', 'limit': 10}
    }
    
    valid, error = validator.validate_request(test_request)
    print(f"Request validation: {'Valid' if valid else f'Invalid - {error}'}")


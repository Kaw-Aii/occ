# Hypergraph Database Integration

This directory contains the database persistence layer for the OpenCog Collection's hypergraph memory and cognitive synergy framework.

## Overview

The hypergraph database integration enables persistent storage of cognitive architecture data, including atoms, links, patterns, cognitive processes, and synergy metrics. This persistence layer supports both **Supabase** and **Neon** PostgreSQL databases.

## Architecture

The database schema is designed to support the core concepts of cognitive synergy:

- **Atoms**: Nodes and links in the hypergraph knowledge representation
- **Atom Links**: Directed edges connecting atoms with typed relationships
- **Cognitive Processes**: Active processes participating in cognitive synergy
- **Patterns**: Discovered patterns from pattern mining algorithms
- **Synergy Events**: Log of interactions between cognitive processes
- **Attention Log**: History of attention allocation changes
- **Synergy Metrics**: Time-series performance metrics

## Setup

### Prerequisites

- PostgreSQL 15+ (Supabase or Neon)
- Python 3.11+
- Supabase Python SDK

### Installation

Install the required Python packages:

```bash
pip install supabase pandas numpy
```

### Database Initialization

#### For Supabase

1. Create a new Supabase project at [supabase.com](https://supabase.com)
2. Navigate to the SQL Editor in your project dashboard
3. Copy and paste the contents of `hypergraph_schema.sql`
4. Execute the SQL script to create all tables, indexes, and views
5. Note your project URL and API key from Settings > API

#### For Neon

1. Create a new Neon project at [neon.tech](https://neon.tech)
2. Connect to your database using the provided connection string
3. Execute the schema:

```bash
psql "postgresql://user:password@host/database" -f hypergraph_schema.sql
```

### Environment Configuration

Set the following environment variables:

```bash
export SUPABASE_URL="https://your-project.supabase.co"
export SUPABASE_KEY="your-api-key"
```

Or for Neon, the Supabase client can connect using a connection string.

## Usage

### Python Integration

```python
from database.hypergraph_persistence import HypergraphPersistence

# Initialize persistence layer
persistence = HypergraphPersistence()

# Create an atom
atom_id = persistence.save_atom(
    atom_type='ConceptNode',
    name='cognitive_synergy',
    truth_value=0.95,
    attention_value=0.8,
    metadata={'domain': 'AGI', 'importance': 'high'}
)

# Create a link between atoms
link_id = persistence.create_link(
    source_atom_id=atom1_id,
    target_atom_id=atom2_id,
    link_type='InheritanceLink',
    strength=0.9
)

# Query high-attention atoms
atoms = persistence.find_atoms(min_attention=0.7, limit=50)

# Update attention
persistence.update_attention(
    atom_id=atom_id,
    attention_delta=0.1,
    reason='pattern_discovery'
)

# Register a cognitive process
process_id = persistence.register_process(
    process_name='PatternMiner',
    process_type='mining',
    priority=0.9
)

# Log a synergy event
event_id = persistence.log_synergy_event(
    event_type='bottleneck_resolution',
    source_process_id=process1_id,
    target_process_id=process2_id,
    outcome='success'
)

# Record metrics
persistence.record_metric(
    metric_name='process_efficiency',
    metric_value=0.87
)

# Get metrics summary
metrics = persistence.get_metrics_summary(hours=24)
```

### Integration with Cognitive Synergy Framework

```python
from cognitive_synergy_framework import CognitiveSynergyEngine
from database.hypergraph_persistence import HypergraphPersistence

# Initialize with persistence
persistence = HypergraphPersistence()
engine = CognitiveSynergyEngine()

# Sync memory to database
for atom_id, atom in engine.memory.atoms.items():
    persistence.save_atom(
        atom_type=atom.atom_type,
        name=atom.name,
        truth_value=atom.truth_value,
        attention_value=atom.attention_value,
        metadata=atom.metadata
    )

# Sync processes
for process_id, process in engine.processes.items():
    persistence.register_process(
        process_name=process.process_id,
        process_type=process.process_type,
        priority=process.priority
    )
```

## Database Schema

### Core Tables

#### `atoms`
Stores all atoms (nodes and links) in the hypergraph.

| Column | Type | Description |
|--------|------|-------------|
| atom_id | UUID | Primary key |
| atom_type | VARCHAR(100) | Type of atom (e.g., ConceptNode) |
| name | TEXT | Atom name/identifier |
| truth_value | FLOAT | Truth value (0.0-1.0) |
| attention_value | FLOAT | Attention value |
| confidence | FLOAT | Confidence level |
| metadata | JSONB | Additional metadata |
| created_at | TIMESTAMP | Creation timestamp |
| updated_at | TIMESTAMP | Last update timestamp |

#### `atom_links`
Stores directed edges between atoms.

| Column | Type | Description |
|--------|------|-------------|
| link_id | UUID | Primary key |
| source_atom_id | UUID | Source atom reference |
| target_atom_id | UUID | Target atom reference |
| link_type | VARCHAR(100) | Type of link |
| strength | FLOAT | Link strength (0.0-1.0) |
| metadata | JSONB | Additional metadata |

#### `cognitive_processes`
Tracks active cognitive processes.

| Column | Type | Description |
|--------|------|-------------|
| process_id | UUID | Primary key |
| process_name | VARCHAR(200) | Process name |
| process_type | VARCHAR(100) | Process type |
| priority | FLOAT | Priority level |
| status | VARCHAR(50) | Current status |
| is_stuck | BOOLEAN | Bottleneck indicator |
| performance_metrics | JSONB | Performance data |
| last_activity | TIMESTAMP | Last activity time |

### Views

The schema includes several materialized views for efficient querying:

- `high_attention_atoms`: Atoms with attention > 0.5
- `active_processes`: Currently active cognitive processes
- `recent_synergy_events`: Last 100 synergy events
- `synergy_metrics_summary`: 24-hour metrics aggregation

## MCP Server Integration

The AtomSpace MCP server (`mcp/atomspace_mcp_server.py`) provides external access to the hypergraph through the Model Context Protocol:

```bash
# Run the MCP server
python mcp/atomspace_mcp_server.py
```

Available MCP tools:
- `query_atoms`: Query atoms by type and attention
- `create_atom`: Create new atoms
- `link_atoms`: Create links between atoms
- `get_atom_neighbors`: Get connected atoms
- `allocate_attention`: Update attention values
- `get_synergy_metrics`: Retrieve metrics
- `find_patterns`: Search discovered patterns
- `get_active_processes`: List cognitive processes

## Performance Considerations

The schema includes optimized indexes for common query patterns:

- B-tree indexes on frequently queried columns
- GIN indexes for JSONB metadata fields
- Composite indexes for multi-column queries
- Automatic timestamp updates via triggers

For large-scale deployments, consider:

- Partitioning the `synergy_metrics` table by time
- Implementing connection pooling
- Using read replicas for query-heavy workloads
- Enabling query result caching

## Security

Row Level Security (RLS) is enabled on all tables with policies that:

- Allow authenticated users to read all data
- Allow authenticated users to insert/update data
- Prevent unauthorized access

For production deployments, customize RLS policies based on your security requirements.

## Monitoring

Query the `synergy_metrics_summary` view to monitor system health:

```sql
SELECT * FROM synergy_metrics_summary
ORDER BY last_recorded DESC;
```

Track attention distribution:

```sql
SELECT 
    atom_type,
    COUNT(*) as count,
    AVG(attention_value) as avg_attention,
    MAX(attention_value) as max_attention
FROM atoms
GROUP BY atom_type
ORDER BY avg_attention DESC;
```

## Contributing

When extending the schema:

1. Add new tables/columns with appropriate indexes
2. Update the `HypergraphPersistence` class with new methods
3. Add corresponding MCP tools if external access is needed
4. Update this README with usage examples
5. Test thoroughly with the CI pipeline

## License

GPL-3.0+ - See the main repository LICENSE file for details.


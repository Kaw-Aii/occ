-- Hypergraph Database Schema for OpenCog Collection
-- Designed for Supabase/Neon PostgreSQL
-- Purpose: Persist AtomSpace hypergraph data for cognitive synergy

-- Enable UUID extension
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Atoms table: Core hypergraph nodes and links
CREATE TABLE IF NOT EXISTS atoms (
    atom_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    atom_type VARCHAR(100) NOT NULL,
    name TEXT NOT NULL,
    truth_value FLOAT DEFAULT 1.0 CHECK (truth_value >= 0 AND truth_value <= 1),
    attention_value FLOAT DEFAULT 0.0,
    confidence FLOAT DEFAULT 1.0 CHECK (confidence >= 0 AND confidence <= 1),
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    UNIQUE(atom_type, name)
);

-- Atom links: Hypergraph edges connecting atoms
CREATE TABLE IF NOT EXISTS atom_links (
    link_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    source_atom_id UUID NOT NULL REFERENCES atoms(atom_id) ON DELETE CASCADE,
    target_atom_id UUID NOT NULL REFERENCES atoms(atom_id) ON DELETE CASCADE,
    link_type VARCHAR(100) NOT NULL,
    strength FLOAT DEFAULT 1.0 CHECK (strength >= 0 AND strength <= 1),
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    UNIQUE(source_atom_id, target_atom_id, link_type)
);

-- Cognitive processes: Track active cognitive processes
CREATE TABLE IF NOT EXISTS cognitive_processes (
    process_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    process_name VARCHAR(200) NOT NULL,
    process_type VARCHAR(100) NOT NULL,
    priority FLOAT DEFAULT 1.0,
    status VARCHAR(50) DEFAULT 'active',
    is_stuck BOOLEAN DEFAULT FALSE,
    bottleneck_threshold FLOAT DEFAULT 0.8,
    performance_metrics JSONB DEFAULT '{}',
    last_activity TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Patterns: Discovered patterns from pattern mining
CREATE TABLE IF NOT EXISTS patterns (
    pattern_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    pattern_type VARCHAR(100) NOT NULL,
    pattern_data JSONB NOT NULL,
    frequency INTEGER DEFAULT 1,
    confidence FLOAT DEFAULT 0.5,
    support FLOAT DEFAULT 0.5,
    discovered_by UUID REFERENCES cognitive_processes(process_id),
    discovered_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    last_seen TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Synergy events: Track cognitive synergy interactions
CREATE TABLE IF NOT EXISTS synergy_events (
    event_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    event_type VARCHAR(100) NOT NULL,
    source_process_id UUID REFERENCES cognitive_processes(process_id),
    target_process_id UUID REFERENCES cognitive_processes(process_id),
    event_data JSONB DEFAULT '{}',
    outcome VARCHAR(50),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Attention allocation log: Track attention distribution
CREATE TABLE IF NOT EXISTS attention_log (
    log_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    atom_id UUID NOT NULL REFERENCES atoms(atom_id) ON DELETE CASCADE,
    attention_delta FLOAT NOT NULL,
    reason TEXT,
    allocated_by UUID REFERENCES cognitive_processes(process_id),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Metrics: System-wide cognitive synergy metrics
CREATE TABLE IF NOT EXISTS synergy_metrics (
    metric_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    metric_name VARCHAR(100) NOT NULL,
    metric_value FLOAT NOT NULL,
    metric_metadata JSONB DEFAULT '{}',
    recorded_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Indexes for performance
CREATE INDEX idx_atoms_type ON atoms(atom_type);
CREATE INDEX idx_atoms_attention ON atoms(attention_value DESC);
CREATE INDEX idx_atoms_created ON atoms(created_at DESC);
CREATE INDEX idx_atom_links_source ON atom_links(source_atom_id);
CREATE INDEX idx_atom_links_target ON atom_links(target_atom_id);
CREATE INDEX idx_atom_links_type ON atom_links(link_type);
CREATE INDEX idx_processes_status ON cognitive_processes(status);
CREATE INDEX idx_processes_priority ON cognitive_processes(priority DESC);
CREATE INDEX idx_patterns_type ON patterns(pattern_type);
CREATE INDEX idx_patterns_frequency ON patterns(frequency DESC);
CREATE INDEX idx_synergy_events_type ON synergy_events(event_type);
CREATE INDEX idx_synergy_events_created ON synergy_events(created_at DESC);
CREATE INDEX idx_metrics_name ON synergy_metrics(metric_name);
CREATE INDEX idx_metrics_recorded ON synergy_metrics(recorded_at DESC);

-- GIN indexes for JSONB columns
CREATE INDEX idx_atoms_metadata ON atoms USING GIN(metadata);
CREATE INDEX idx_atom_links_metadata ON atom_links USING GIN(metadata);
CREATE INDEX idx_patterns_data ON patterns USING GIN(pattern_data);
CREATE INDEX idx_synergy_events_data ON synergy_events USING GIN(event_data);

-- Update timestamp trigger function
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Apply update trigger to atoms table
CREATE TRIGGER update_atoms_updated_at
    BEFORE UPDATE ON atoms
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- Views for common queries

-- High attention atoms view
CREATE OR REPLACE VIEW high_attention_atoms AS
SELECT 
    atom_id,
    atom_type,
    name,
    attention_value,
    truth_value,
    created_at
FROM atoms
WHERE attention_value > 0.5
ORDER BY attention_value DESC;

-- Active cognitive processes view
CREATE OR REPLACE VIEW active_processes AS
SELECT 
    process_id,
    process_name,
    process_type,
    priority,
    is_stuck,
    last_activity
FROM cognitive_processes
WHERE status = 'active'
ORDER BY priority DESC, last_activity DESC;

-- Recent synergy events view
CREATE OR REPLACE VIEW recent_synergy_events AS
SELECT 
    e.event_id,
    e.event_type,
    sp.process_name as source_process,
    tp.process_name as target_process,
    e.outcome,
    e.created_at
FROM synergy_events e
LEFT JOIN cognitive_processes sp ON e.source_process_id = sp.process_id
LEFT JOIN cognitive_processes tp ON e.target_process_id = tp.process_id
ORDER BY e.created_at DESC
LIMIT 100;

-- Synergy metrics summary view
CREATE OR REPLACE VIEW synergy_metrics_summary AS
SELECT 
    metric_name,
    AVG(metric_value) as avg_value,
    MAX(metric_value) as max_value,
    MIN(metric_value) as min_value,
    COUNT(*) as sample_count,
    MAX(recorded_at) as last_recorded
FROM synergy_metrics
WHERE recorded_at > NOW() - INTERVAL '24 hours'
GROUP BY metric_name
ORDER BY metric_name;

-- Row Level Security (RLS) policies
ALTER TABLE atoms ENABLE ROW LEVEL SECURITY;
ALTER TABLE atom_links ENABLE ROW LEVEL SECURITY;
ALTER TABLE cognitive_processes ENABLE ROW LEVEL SECURITY;
ALTER TABLE patterns ENABLE ROW LEVEL SECURITY;
ALTER TABLE synergy_events ENABLE ROW LEVEL SECURITY;
ALTER TABLE attention_log ENABLE ROW LEVEL SECURITY;
ALTER TABLE synergy_metrics ENABLE ROW LEVEL SECURITY;

-- Allow authenticated users to read all data
CREATE POLICY "Allow authenticated read access" ON atoms FOR SELECT TO authenticated USING (true);
CREATE POLICY "Allow authenticated read access" ON atom_links FOR SELECT TO authenticated USING (true);
CREATE POLICY "Allow authenticated read access" ON cognitive_processes FOR SELECT TO authenticated USING (true);
CREATE POLICY "Allow authenticated read access" ON patterns FOR SELECT TO authenticated USING (true);
CREATE POLICY "Allow authenticated read access" ON synergy_events FOR SELECT TO authenticated USING (true);
CREATE POLICY "Allow authenticated read access" ON attention_log FOR SELECT TO authenticated USING (true);
CREATE POLICY "Allow authenticated read access" ON synergy_metrics FOR SELECT TO authenticated USING (true);

-- Allow authenticated users to insert/update data
CREATE POLICY "Allow authenticated write access" ON atoms FOR INSERT TO authenticated WITH CHECK (true);
CREATE POLICY "Allow authenticated update access" ON atoms FOR UPDATE TO authenticated USING (true);
CREATE POLICY "Allow authenticated write access" ON atom_links FOR INSERT TO authenticated WITH CHECK (true);
CREATE POLICY "Allow authenticated write access" ON cognitive_processes FOR ALL TO authenticated USING (true);
CREATE POLICY "Allow authenticated write access" ON patterns FOR ALL TO authenticated USING (true);
CREATE POLICY "Allow authenticated write access" ON synergy_events FOR INSERT TO authenticated WITH CHECK (true);
CREATE POLICY "Allow authenticated write access" ON attention_log FOR INSERT TO authenticated WITH CHECK (true);
CREATE POLICY "Allow authenticated write access" ON synergy_metrics FOR INSERT TO authenticated WITH CHECK (true);

-- Comments for documentation
COMMENT ON TABLE atoms IS 'Core hypergraph nodes and links representing knowledge atoms';
COMMENT ON TABLE atom_links IS 'Directed edges connecting atoms in the hypergraph';
COMMENT ON TABLE cognitive_processes IS 'Active cognitive processes participating in synergy';
COMMENT ON TABLE patterns IS 'Discovered patterns from pattern mining algorithms';
COMMENT ON TABLE synergy_events IS 'Log of cognitive synergy interactions between processes';
COMMENT ON TABLE attention_log IS 'History of attention allocation changes';
COMMENT ON TABLE synergy_metrics IS 'Time-series metrics measuring cognitive synergy effectiveness';


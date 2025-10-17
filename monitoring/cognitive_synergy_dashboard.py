#!/usr/bin/env python3
"""
Cognitive Synergy Monitoring Dashboard
======================================

Real-time monitoring dashboard for the OpenCog Collection cognitive architecture.
Tracks hypergraph metrics, process efficiency, attention distribution, and synergy events.

Features:
- Real-time metrics visualization
- Process health monitoring
- Attention allocation tracking
- Pattern discovery analytics
- Synergy event timeline

Author: OpenCog Collection Contributors
License: GPL-3.0+
"""

import os
import sys
from datetime import datetime, timedelta
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from flask import Flask, render_template_string, jsonify
import threading
import time

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from database.hypergraph_persistence import HypergraphPersistence

# Initialize Flask app
app = Flask(__name__)

# Initialize persistence layer
try:
    persistence = HypergraphPersistence()
    DB_AVAILABLE = True
except Exception as e:
    print(f"Warning: Database not available: {e}")
    DB_AVAILABLE = False

# Dashboard HTML template
DASHBOARD_HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>Cognitive Synergy Dashboard - OpenCog Collection</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: #ffffff;
        }
        .header {
            text-align: center;
            padding: 20px;
            background: rgba(0, 0, 0, 0.3);
            border-radius: 10px;
            margin-bottom: 30px;
        }
        h1 {
            margin: 0;
            font-size: 2.5em;
            text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.5);
        }
        .subtitle {
            font-size: 1.2em;
            opacity: 0.9;
            margin-top: 10px;
        }
        .metrics-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }
        .metric-card {
            background: rgba(255, 255, 255, 0.1);
            backdrop-filter: blur(10px);
            padding: 20px;
            border-radius: 10px;
            border: 1px solid rgba(255, 255, 255, 0.2);
            box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.37);
        }
        .metric-value {
            font-size: 2.5em;
            font-weight: bold;
            margin: 10px 0;
        }
        .metric-label {
            font-size: 1em;
            opacity: 0.8;
            text-transform: uppercase;
            letter-spacing: 1px;
        }
        .chart-container {
            background: rgba(255, 255, 255, 0.95);
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 20px;
            box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.37);
        }
        .chart-title {
            color: #333;
            font-size: 1.5em;
            margin-bottom: 15px;
            font-weight: bold;
        }
        .status-indicator {
            display: inline-block;
            width: 12px;
            height: 12px;
            border-radius: 50%;
            margin-right: 8px;
        }
        .status-active { background-color: #10b981; }
        .status-warning { background-color: #f59e0b; }
        .status-error { background-color: #ef4444; }
        .refresh-info {
            text-align: center;
            opacity: 0.7;
            margin-top: 20px;
            font-size: 0.9em;
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>🧠 Cognitive Synergy Dashboard</h1>
        <div class="subtitle">OpenCog Collection - Real-time Monitoring</div>
    </div>
    
    <div class="metrics-grid" id="metrics-grid">
        <!-- Metrics will be populated by JavaScript -->
    </div>
    
    <div class="chart-container">
        <div class="chart-title">Attention Distribution</div>
        <div id="attention-chart"></div>
    </div>
    
    <div class="chart-container">
        <div class="chart-title">Process Efficiency Over Time</div>
        <div id="efficiency-chart"></div>
    </div>
    
    <div class="chart-container">
        <div class="chart-title">Synergy Events Timeline</div>
        <div id="events-chart"></div>
    </div>
    
    <div class="chart-container">
        <div class="chart-title">Active Cognitive Processes</div>
        <div id="processes-chart"></div>
    </div>
    
    <div class="refresh-info">
        Dashboard auto-refreshes every 30 seconds | Last update: <span id="last-update"></span>
    </div>
    
    <script>
        function updateDashboard() {
            fetch('/api/metrics')
                .then(response => response.json())
                .then(data => {
                    updateMetrics(data.summary);
                    updateCharts(data);
                    document.getElementById('last-update').textContent = new Date().toLocaleTimeString();
                })
                .catch(error => console.error('Error fetching metrics:', error));
        }
        
        function updateMetrics(summary) {
            const metricsHtml = `
                <div class="metric-card">
                    <div class="metric-label">Total Atoms</div>
                    <div class="metric-value">${summary.total_atoms}</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Active Processes</div>
                    <div class="metric-value">${summary.active_processes}</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Process Efficiency</div>
                    <div class="metric-value">${(summary.process_efficiency * 100).toFixed(1)}%</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Pattern Diversity</div>
                    <div class="metric-value">${summary.pattern_diversity}</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">High Attention Atoms</div>
                    <div class="metric-value">${summary.high_attention_count}</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Synergy Events (24h)</div>
                    <div class="metric-value">${summary.synergy_events_24h}</div>
                </div>
            `;
            document.getElementById('metrics-grid').innerHTML = metricsHtml;
        }
        
        function updateCharts(data) {
            // Attention distribution chart
            if (data.attention_distribution) {
                const attentionTrace = {
                    x: data.attention_distribution.map(d => d.atom_type),
                    y: data.attention_distribution.map(d => d.avg_attention),
                    type: 'bar',
                    marker: { color: '#667eea' }
                };
                Plotly.newPlot('attention-chart', [attentionTrace], {
                    margin: { t: 0, b: 80 },
                    xaxis: { title: 'Atom Type' },
                    yaxis: { title: 'Average Attention' }
                });
            }
            
            // Process efficiency timeline
            if (data.efficiency_timeline) {
                const efficiencyTrace = {
                    x: data.efficiency_timeline.map(d => d.time),
                    y: data.efficiency_timeline.map(d => d.efficiency),
                    type: 'scatter',
                    mode: 'lines+markers',
                    line: { color: '#10b981', width: 3 }
                };
                Plotly.newPlot('efficiency-chart', [efficiencyTrace], {
                    margin: { t: 0, b: 80 },
                    xaxis: { title: 'Time' },
                    yaxis: { title: 'Efficiency', range: [0, 1] }
                });
            }
            
            // Synergy events
            if (data.synergy_events) {
                const eventsTrace = {
                    x: data.synergy_events.map(d => d.time),
                    y: data.synergy_events.map(d => d.event_type),
                    type: 'scatter',
                    mode: 'markers',
                    marker: { size: 10, color: '#f59e0b' }
                };
                Plotly.newPlot('events-chart', [eventsTrace], {
                    margin: { t: 0, b: 80 },
                    xaxis: { title: 'Time' },
                    yaxis: { title: 'Event Type' }
                });
            }
            
            // Active processes
            if (data.active_processes) {
                const processesTrace = {
                    labels: data.active_processes.map(d => d.process_name),
                    values: data.active_processes.map(d => d.priority),
                    type: 'pie',
                    marker: { colors: ['#667eea', '#764ba2', '#f59e0b', '#10b981', '#ef4444'] }
                };
                Plotly.newPlot('processes-chart', [processesTrace], {
                    margin: { t: 0, b: 0 }
                });
            }
        }
        
        // Initial load
        updateDashboard();
        
        // Auto-refresh every 30 seconds
        setInterval(updateDashboard, 30000);
    </script>
</body>
</html>
"""


@app.route('/')
def dashboard():
    """Render the main dashboard."""
    return render_template_string(DASHBOARD_HTML)


@app.route('/api/metrics')
def get_metrics():
    """API endpoint for fetching current metrics."""
    if not DB_AVAILABLE:
        return jsonify({
            'summary': {
                'total_atoms': 0,
                'active_processes': 0,
                'process_efficiency': 0.0,
                'pattern_diversity': 0,
                'high_attention_count': 0,
                'synergy_events_24h': 0
            },
            'attention_distribution': [],
            'efficiency_timeline': [],
            'synergy_events': [],
            'active_processes': []
        })
    
    try:
        # Get summary metrics
        atoms_result = persistence.client.table('atoms').select('*', count='exact').execute()
        total_atoms = atoms_result.count if hasattr(atoms_result, 'count') else len(atoms_result.data)
        
        processes_result = persistence.client.table('cognitive_processes').select('*').eq('status', 'active').execute()
        active_processes = len(processes_result.data)
        
        high_attention_result = persistence.client.table('atoms').select('*').gte('attention_value', 0.5).execute()
        high_attention_count = len(high_attention_result.data)
        
        patterns_result = persistence.client.table('patterns').select('*', count='exact').execute()
        pattern_diversity = patterns_result.count if hasattr(patterns_result, 'count') else len(patterns_result.data)
        
        # Get recent synergy events
        events_result = persistence.client.table('synergy_events').select('*').gte(
            'created_at', (datetime.utcnow() - timedelta(hours=24)).isoformat()
        ).execute()
        synergy_events_24h = len(events_result.data)
        
        # Get attention distribution by atom type
        attention_dist_query = """
        SELECT atom_type, AVG(attention_value) as avg_attention, COUNT(*) as count
        FROM atoms
        GROUP BY atom_type
        ORDER BY avg_attention DESC
        LIMIT 10
        """
        # Note: Direct SQL queries require RPC or custom function in Supabase
        # For now, we'll aggregate in Python
        atoms_data = atoms_result.data
        attention_by_type = {}
        for atom in atoms_data:
            atom_type = atom['atom_type']
            if atom_type not in attention_by_type:
                attention_by_type[atom_type] = []
            attention_by_type[atom_type].append(atom['attention_value'])
        
        attention_distribution = [
            {
                'atom_type': atom_type,
                'avg_attention': sum(values) / len(values) if values else 0,
                'count': len(values)
            }
            for atom_type, values in attention_by_type.items()
        ]
        attention_distribution.sort(key=lambda x: x['avg_attention'], reverse=True)
        attention_distribution = attention_distribution[:10]
        
        # Get efficiency timeline (last 24 hours)
        metrics_result = persistence.client.table('synergy_metrics').select('*').eq(
            'metric_name', 'process_efficiency'
        ).gte(
            'recorded_at', (datetime.utcnow() - timedelta(hours=24)).isoformat()
        ).order('recorded_at').execute()
        
        efficiency_timeline = [
            {
                'time': m['recorded_at'],
                'efficiency': m['metric_value']
            }
            for m in metrics_result.data
        ]
        
        # Get recent synergy events for timeline
        synergy_events = [
            {
                'time': e['created_at'],
                'event_type': e['event_type'],
                'outcome': e.get('outcome', 'unknown')
            }
            for e in events_result.data[:50]
        ]
        
        # Get active processes info
        active_processes_data = [
            {
                'process_name': p['process_name'],
                'process_type': p['process_type'],
                'priority': p['priority'],
                'is_stuck': p['is_stuck']
            }
            for p in processes_result.data
        ]
        
        # Calculate process efficiency
        stuck_count = sum(1 for p in processes_result.data if p.get('is_stuck', False))
        process_efficiency = (active_processes - stuck_count) / active_processes if active_processes > 0 else 0.0
        
        return jsonify({
            'summary': {
                'total_atoms': total_atoms,
                'active_processes': active_processes,
                'process_efficiency': process_efficiency,
                'pattern_diversity': pattern_diversity,
                'high_attention_count': high_attention_count,
                'synergy_events_24h': synergy_events_24h
            },
            'attention_distribution': attention_distribution,
            'efficiency_timeline': efficiency_timeline,
            'synergy_events': synergy_events,
            'active_processes': active_processes_data
        })
    
    except Exception as e:
        print(f"Error fetching metrics: {e}")
        return jsonify({'error': str(e)}), 500


def run_dashboard(host='0.0.0.0', port=8050, debug=False):
    """
    Run the dashboard server.
    
    Args:
        host: Host address to bind to
        port: Port number to listen on
        debug: Enable debug mode
    """
    print(f"Starting Cognitive Synergy Dashboard on http://{host}:{port}")
    app.run(host=host, port=port, debug=debug)


if __name__ == '__main__':
    run_dashboard(debug=True)


"""
Cognitive Monitoring and Observability for OpenCog Collection
=============================================================

This module provides monitoring, observability, and analytics for the
cognitive architecture, enabling real-time tracking of cognitive synergy
metrics and system health.

Features:
- Real-time metrics collection and aggregation
- Cognitive synergy dashboard data generation
- Performance tracking and bottleneck detection
- Historical trend analysis
- Alert generation for anomalies
- Export capabilities for external monitoring systems

Author: OpenCog Collection Contributors
License: GPL-3.0+
"""

import numpy as np
import json
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from collections import deque, defaultdict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class MetricSnapshot:
    """Single point-in-time metric snapshot."""
    timestamp: datetime
    metric_name: str
    value: float
    tags: Dict[str, str] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'timestamp': self.timestamp.isoformat(),
            'metric_name': self.metric_name,
            'value': self.value,
            'tags': self.tags
        }


@dataclass
class Alert:
    """Alert for anomalous conditions."""
    alert_id: str
    severity: str  # 'info', 'warning', 'critical'
    component: str
    message: str
    timestamp: datetime = field(default_factory=datetime.now)
    resolved: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'alert_id': self.alert_id,
            'severity': self.severity,
            'component': self.component,
            'message': self.message,
            'timestamp': self.timestamp.isoformat(),
            'resolved': self.resolved
        }


class MetricsCollector:
    """Collects and aggregates metrics from cognitive components."""
    
    def __init__(self, retention_hours: int = 24):
        """
        Initialize metrics collector.
        
        Args:
            retention_hours: Hours to retain metric history
        """
        self.metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=10000))
        self.retention_hours = retention_hours
        self.last_cleanup = datetime.now()
        
        logger.info(f"Metrics Collector initialized (retention={retention_hours}h)")
    
    def record_metric(self, metric_name: str, value: float, 
                     tags: Optional[Dict[str, str]] = None):
        """Record a metric value."""
        snapshot = MetricSnapshot(
            timestamp=datetime.now(),
            metric_name=metric_name,
            value=value,
            tags=tags or {}
        )
        
        self.metrics[metric_name].append(snapshot)
        logger.debug(f"Recorded metric: {metric_name}={value}")
    
    def get_metric_history(self, metric_name: str, 
                          hours: Optional[int] = None) -> List[MetricSnapshot]:
        """Get historical values for a metric."""
        if metric_name not in self.metrics:
            return []
        
        snapshots = list(self.metrics[metric_name])
        
        if hours is not None:
            cutoff = datetime.now() - timedelta(hours=hours)
            snapshots = [s for s in snapshots if s.timestamp >= cutoff]
        
        return snapshots
    
    def get_metric_stats(self, metric_name: str, 
                        hours: Optional[int] = None) -> Dict[str, float]:
        """Get statistical summary of a metric."""
        history = self.get_metric_history(metric_name, hours)
        
        if not history:
            return {
                'count': 0,
                'mean': 0.0,
                'std': 0.0,
                'min': 0.0,
                'max': 0.0
            }
        
        values = [s.value for s in history]
        
        return {
            'count': len(values),
            'mean': float(np.mean(values)),
            'std': float(np.std(values)),
            'min': float(np.min(values)),
            'max': float(np.max(values)),
            'latest': values[-1] if values else 0.0
        }
    
    def cleanup_old_metrics(self):
        """Remove metrics older than retention period."""
        cutoff = datetime.now() - timedelta(hours=self.retention_hours)
        
        for metric_name in self.metrics:
            snapshots = self.metrics[metric_name]
            # Remove old snapshots
            while snapshots and snapshots[0].timestamp < cutoff:
                snapshots.popleft()
        
        self.last_cleanup = datetime.now()
        logger.debug("Old metrics cleaned up")
    
    def export_metrics(self, format: str = 'json') -> str:
        """Export all metrics in specified format."""
        if format == 'json':
            export_data = {
                'timestamp': datetime.now().isoformat(),
                'metrics': {}
            }
            
            for metric_name, snapshots in self.metrics.items():
                export_data['metrics'][metric_name] = [
                    s.to_dict() for s in list(snapshots)[-100:]  # Last 100 points
                ]
            
            return json.dumps(export_data, indent=2)
        
        else:
            raise ValueError(f"Unsupported export format: {format}")


class CognitiveMonitor:
    """
    Main monitoring system for cognitive architecture.
    
    Tracks health, performance, and synergy metrics across all components.
    """
    
    def __init__(self):
        """Initialize cognitive monitor."""
        self.collector = MetricsCollector()
        self.alerts: deque = deque(maxlen=1000)
        self.alert_rules: List[Dict[str, Any]] = []
        
        # Component health tracking
        self.component_health: Dict[str, str] = {}
        
        # Initialize default alert rules
        self._setup_default_alert_rules()
        
        logger.info("Cognitive Monitor initialized")
    
    def _setup_default_alert_rules(self):
        """Set up default alerting rules."""
        self.alert_rules = [
            {
                'name': 'low_synergy_index',
                'metric': 'synergy_index',
                'condition': 'below',
                'threshold': 0.3,
                'severity': 'warning',
                'message': 'Cognitive synergy index is low'
            },
            {
                'name': 'low_aar_coherence',
                'metric': 'aar_coherence',
                'condition': 'below',
                'threshold': 0.4,
                'severity': 'warning',
                'message': 'AAR self-coherence is low'
            },
            {
                'name': 'high_bottleneck_count',
                'metric': 'bottleneck_count',
                'condition': 'above',
                'threshold': 5,
                'severity': 'critical',
                'message': 'High number of bottlenecks detected'
            },
            {
                'name': 'low_cognitive_efficiency',
                'metric': 'cognitive_efficiency',
                'condition': 'below',
                'threshold': 0.5,
                'severity': 'warning',
                'message': 'Cognitive processing efficiency is low'
            }
        ]
    
    def record_component_metrics(self, component: str, metrics: Dict[str, float]):
        """Record metrics from a cognitive component."""
        for metric_name, value in metrics.items():
            full_metric_name = f"{component}.{metric_name}"
            self.collector.record_metric(
                full_metric_name,
                value,
                tags={'component': component}
            )
        
        # Update component health
        self._update_component_health(component, metrics)
    
    def _update_component_health(self, component: str, metrics: Dict[str, float]):
        """Update health status for a component."""
        # Simple heuristic: check if key metrics are in healthy range
        health_score = 0.0
        metric_count = 0
        
        for metric_name, value in metrics.items():
            if 'coherence' in metric_name or 'efficiency' in metric_name:
                health_score += value
                metric_count += 1
        
        if metric_count > 0:
            avg_health = health_score / metric_count
            
            if avg_health > 0.7:
                self.component_health[component] = 'healthy'
            elif avg_health > 0.4:
                self.component_health[component] = 'degraded'
            else:
                self.component_health[component] = 'unhealthy'
        else:
            self.component_health[component] = 'unknown'
    
    def check_alerts(self):
        """Check alert rules and generate alerts if needed."""
        for rule in self.alert_rules:
            metric_name = rule['metric']
            stats = self.collector.get_metric_stats(metric_name, hours=1)
            
            if stats['count'] == 0:
                continue
            
            latest_value = stats['latest']
            threshold = rule['threshold']
            condition = rule['condition']
            
            triggered = False
            
            if condition == 'above' and latest_value > threshold:
                triggered = True
            elif condition == 'below' and latest_value < threshold:
                triggered = True
            
            if triggered:
                alert = Alert(
                    alert_id=f"{rule['name']}_{int(datetime.now().timestamp())}",
                    severity=rule['severity'],
                    component=metric_name.split('.')[0] if '.' in metric_name else 'system',
                    message=f"{rule['message']} (value={latest_value:.3f}, threshold={threshold})"
                )
                
                self.alerts.append(alert)
                logger.warning(f"Alert triggered: {alert.message}")
    
    def get_dashboard_data(self) -> Dict[str, Any]:
        """Generate data for monitoring dashboard."""
        # Get key metrics
        synergy_stats = self.collector.get_metric_stats('synergy_index', hours=1)
        aar_stats = self.collector.get_metric_stats('aar_coherence', hours=1)
        hg_stats = self.collector.get_metric_stats('hypergraph_connectivity', hours=1)
        
        # Get recent alerts
        recent_alerts = [a for a in self.alerts if not a.resolved][-10:]
        
        dashboard = {
            'timestamp': datetime.now().isoformat(),
            'system_health': self._compute_overall_health(),
            'components': {
                name: status for name, status in self.component_health.items()
            },
            'key_metrics': {
                'synergy_index': synergy_stats,
                'aar_coherence': aar_stats,
                'hypergraph_connectivity': hg_stats
            },
            'alerts': {
                'active': len(recent_alerts),
                'recent': [a.to_dict() for a in recent_alerts]
            },
            'trends': self._compute_trends()
        }
        
        return dashboard
    
    def _compute_overall_health(self) -> str:
        """Compute overall system health status."""
        if not self.component_health:
            return 'unknown'
        
        health_scores = {
            'healthy': 1.0,
            'degraded': 0.5,
            'unhealthy': 0.0,
            'unknown': 0.5
        }
        
        avg_score = np.mean([
            health_scores.get(status, 0.5) 
            for status in self.component_health.values()
        ])
        
        if avg_score > 0.8:
            return 'healthy'
        elif avg_score > 0.5:
            return 'degraded'
        else:
            return 'unhealthy'
    
    def _compute_trends(self) -> Dict[str, str]:
        """Compute trend direction for key metrics."""
        trends = {}
        
        for metric_name in ['synergy_index', 'aar_coherence', 'hypergraph_connectivity']:
            history = self.collector.get_metric_history(metric_name, hours=1)
            
            if len(history) < 2:
                trends[metric_name] = 'stable'
                continue
            
            # Compare recent average to older average
            values = [s.value for s in history]
            mid_point = len(values) // 2
            
            recent_avg = np.mean(values[mid_point:])
            older_avg = np.mean(values[:mid_point])
            
            diff = recent_avg - older_avg
            
            if abs(diff) < 0.05:
                trends[metric_name] = 'stable'
            elif diff > 0:
                trends[metric_name] = 'improving'
            else:
                trends[metric_name] = 'declining'
        
        return trends
    
    def export_monitoring_data(self, include_history: bool = True) -> Dict[str, Any]:
        """Export complete monitoring data."""
        export = {
            'timestamp': datetime.now().isoformat(),
            'dashboard': self.get_dashboard_data(),
            'alerts': [a.to_dict() for a in self.alerts],
            'component_health': self.component_health
        }
        
        if include_history:
            export['metrics_history'] = json.loads(self.collector.export_metrics())
        
        return export
    
    def save_to_file(self, filepath: str):
        """Save monitoring data to file."""
        data = self.export_monitoring_data(include_history=True)
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Monitoring data saved to {filepath}")


class CognitiveAnalytics:
    """Advanced analytics for cognitive architecture performance."""
    
    def __init__(self, monitor: CognitiveMonitor):
        """Initialize analytics engine."""
        self.monitor = monitor
        logger.info("Cognitive Analytics initialized")
    
    def analyze_synergy_patterns(self, hours: int = 24) -> Dict[str, Any]:
        """Analyze patterns in cognitive synergy over time."""
        synergy_history = self.monitor.collector.get_metric_history('synergy_index', hours)
        
        if not synergy_history:
            return {'error': 'No synergy data available'}
        
        values = [s.value for s in synergy_history]
        timestamps = [s.timestamp for s in synergy_history]
        
        # Compute statistics
        analysis = {
            'time_range_hours': hours,
            'data_points': len(values),
            'statistics': {
                'mean': float(np.mean(values)),
                'std': float(np.std(values)),
                'min': float(np.min(values)),
                'max': float(np.max(values)),
                'range': float(np.max(values) - np.min(values))
            },
            'trend': self._compute_linear_trend(values),
            'stability': self._compute_stability(values),
            'peak_times': self._find_peaks(timestamps, values)
        }
        
        return analysis
    
    def _compute_linear_trend(self, values: List[float]) -> Dict[str, float]:
        """Compute linear trend in values."""
        if len(values) < 2:
            return {'slope': 0.0, 'direction': 'stable'}
        
        x = np.arange(len(values))
        coeffs = np.polyfit(x, values, 1)
        slope = float(coeffs[0])
        
        direction = 'stable'
        if slope > 0.001:
            direction = 'increasing'
        elif slope < -0.001:
            direction = 'decreasing'
        
        return {
            'slope': slope,
            'direction': direction
        }
    
    def _compute_stability(self, values: List[float]) -> float:
        """Compute stability score (inverse of coefficient of variation)."""
        if not values:
            return 0.0
        
        mean = np.mean(values)
        std = np.std(values)
        
        if mean == 0:
            return 0.0
        
        cv = std / mean
        stability = 1.0 / (1.0 + cv)
        
        return float(stability)
    
    def _find_peaks(self, timestamps: List[datetime], 
                   values: List[float]) -> List[Dict[str, Any]]:
        """Find peak values in time series."""
        if len(values) < 3:
            return []
        
        peaks = []
        
        for i in range(1, len(values) - 1):
            if values[i] > values[i-1] and values[i] > values[i+1]:
                if values[i] > np.mean(values):  # Only significant peaks
                    peaks.append({
                        'timestamp': timestamps[i].isoformat(),
                        'value': values[i]
                    })
        
        return peaks[-5:]  # Return last 5 peaks
    
    def generate_report(self) -> str:
        """Generate comprehensive analytics report."""
        dashboard = self.monitor.get_dashboard_data()
        synergy_analysis = self.analyze_synergy_patterns(hours=24)
        
        report = f"""
Cognitive Architecture Analytics Report
========================================
Generated: {datetime.now().isoformat()}

System Health: {dashboard['system_health'].upper()}

Component Status:
{self._format_component_status(dashboard['components'])}

Key Metrics (Last Hour):
{self._format_key_metrics(dashboard['key_metrics'])}

Synergy Analysis (Last 24 Hours):
{self._format_synergy_analysis(synergy_analysis)}

Active Alerts: {dashboard['alerts']['active']}
{self._format_alerts(dashboard['alerts']['recent'])}

Trends:
{self._format_trends(dashboard['trends'])}
"""
        
        return report
    
    def _format_component_status(self, components: Dict[str, str]) -> str:
        """Format component status for report."""
        lines = []
        for name, status in components.items():
            status_symbol = '✓' if status == 'healthy' else '⚠' if status == 'degraded' else '✗'
            lines.append(f"  {status_symbol} {name}: {status}")
        return '\n'.join(lines) if lines else "  No components"
    
    def _format_key_metrics(self, metrics: Dict[str, Dict[str, float]]) -> str:
        """Format key metrics for report."""
        lines = []
        for name, stats in metrics.items():
            if stats['count'] > 0:
                lines.append(f"  {name}: {stats['latest']:.3f} (mean={stats['mean']:.3f}, std={stats['std']:.3f})")
        return '\n'.join(lines) if lines else "  No metrics"
    
    def _format_synergy_analysis(self, analysis: Dict[str, Any]) -> str:
        """Format synergy analysis for report."""
        if 'error' in analysis:
            return f"  {analysis['error']}"
        
        stats = analysis['statistics']
        trend = analysis['trend']
        
        return f"""  Mean: {stats['mean']:.3f}
  Range: [{stats['min']:.3f}, {stats['max']:.3f}]
  Trend: {trend['direction']} (slope={trend['slope']:.6f})
  Stability: {analysis['stability']:.3f}
  Peaks found: {len(analysis['peak_times'])}"""
    
    def _format_alerts(self, alerts: List[Dict[str, Any]]) -> str:
        """Format alerts for report."""
        if not alerts:
            return "  No active alerts"
        
        lines = []
        for alert in alerts:
            severity_symbol = '🔴' if alert['severity'] == 'critical' else '🟡'
            lines.append(f"  {severity_symbol} [{alert['severity']}] {alert['message']}")
        
        return '\n'.join(lines)
    
    def _format_trends(self, trends: Dict[str, str]) -> str:
        """Format trends for report."""
        lines = []
        for metric, trend in trends.items():
            trend_symbol = '↑' if trend == 'improving' else '↓' if trend == 'declining' else '→'
            lines.append(f"  {trend_symbol} {metric}: {trend}")
        return '\n'.join(lines) if lines else "  No trends"


# Example usage and testing
if __name__ == "__main__":
    print("=== Cognitive Monitoring Module Test ===\n")
    
    # Create monitor
    monitor = CognitiveMonitor()
    
    # Simulate metric recording
    print("Recording test metrics...")
    for i in range(20):
        monitor.record_component_metrics('aar', {
            'coherence': 0.7 + 0.1 * np.sin(i * 0.5),
            'stability': 0.8 + 0.05 * np.random.randn()
        })
        
        monitor.record_component_metrics('hypergraph', {
            'connectivity': 0.6 + 0.1 * np.cos(i * 0.3),
            'pattern_count': 10 + i
        })
        
        monitor.collector.record_metric('synergy_index', 
                                       0.65 + 0.15 * np.sin(i * 0.4))
    
    # Check alerts
    monitor.check_alerts()
    
    # Get dashboard
    dashboard = monitor.get_dashboard_data()
    print(f"\nSystem Health: {dashboard['system_health']}")
    print(f"Active Alerts: {dashboard['alerts']['active']}")
    
    # Analytics
    analytics = CognitiveAnalytics(monitor)
    report = analytics.generate_report()
    print(report)
    
    # Export
    export_file = '/tmp/cognitive_monitoring_test.json'
    monitor.save_to_file(export_file)
    print(f"\n✓ Monitoring data exported to {export_file}")
    
    print("\n✓ Cognitive Monitoring module test complete")

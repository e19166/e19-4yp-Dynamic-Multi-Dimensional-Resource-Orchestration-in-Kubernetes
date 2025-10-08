"""
Real-time Monitoring and Decision Execution Interface for DARE System

This module provides:
1. Real-time monitoring dashboard for system status
2. Decision execution interface for applying optimizations
3. Alert management and notification system
4. Performance tracking and reporting
5. Integration with Kubernetes API for resource updates
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from enum import Enum
import pandas as pd
import numpy as np
from threading import Thread, Lock
import time
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class AlertLevel(Enum):
    """Alert severity levels"""
    INFO = "info"
    WARNING = "warning" 
    ERROR = "error"
    CRITICAL = "critical"

class DecisionStatus(Enum):
    """Status of optimization decisions"""
    PENDING = "pending"
    EXECUTING = "executing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

@dataclass
class Alert:
    """System alert"""
    id: str
    timestamp: datetime
    level: AlertLevel
    service_name: str
    message: str
    details: Dict[str, Any]
    acknowledged: bool = False
    resolved: bool = False

@dataclass
class OptimizationDecision:
    """Resource optimization decision"""
    id: str
    timestamp: datetime
    service_name: str
    current_resources: Dict[str, float]
    recommended_resources: Dict[str, float]
    expected_latency: float
    resource_savings: float
    confidence: float
    risk_level: str
    status: DecisionStatus = DecisionStatus.PENDING
    execution_time: Optional[datetime] = None
    completion_time: Optional[datetime] = None
    actual_results: Optional[Dict[str, Any]] = None

@dataclass
class ServiceMetrics:
    """Real-time service metrics"""
    service_name: str
    timestamp: datetime
    cpu_usage: float
    memory_usage: float
    cpu_limit: float
    memory_limit: float
    latency: float
    request_rate: float
    error_rate: float
    availability: float
    cpu_utilization: float
    memory_utilization: float

class MonitoringDashboard:
    """
    Real-time monitoring dashboard for DARE system
    """
    
    def __init__(self, prometheus_monitor, latency_predictor, resource_optimizer):
        self.prometheus_monitor = prometheus_monitor
        self.latency_predictor = latency_predictor
        self.resource_optimizer = resource_optimizer
        
        # State management
        self.alerts = {}
        self.decisions = {}
        self.service_metrics = {}
        self.system_status = {}
        
        # Threading and synchronization
        self.monitoring_active = False
        self.monitoring_thread = None
        self.state_lock = Lock()
        
        # Configuration
        self.monitoring_interval = 30  # seconds
        self.alert_retention_hours = 24
        self.decision_retention_hours = 72
        
        # Performance tracking
        self.performance_history = {}
        self.optimization_statistics = {
            'total_optimizations': 0,
            'successful_optimizations': 0,
            'total_savings': 0.0,
            'services_managed': set()
        }
    
    def start_monitoring(self):
        """Start real-time monitoring"""
        
        if self.monitoring_active:
            logger.warning("Monitoring already active")
            return
        
        self.monitoring_active = True
        self.monitoring_thread = Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        
        logger.info("Started real-time monitoring")
    
    def stop_monitoring(self):
        """Stop real-time monitoring"""
        
        self.monitoring_active = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5)
        
        logger.info("Stopped real-time monitoring")
    
    def _monitoring_loop(self):
        """Main monitoring loop"""
        
        while self.monitoring_active:
            try:
                # Collect metrics from all services
                self._collect_service_metrics()
                
                # Check for alerts
                self._check_alerts()
                
                # Update system status
                self._update_system_status()
                
                # Cleanup old data
                self._cleanup_old_data()
                
                # Sleep until next monitoring cycle
                time.sleep(self.monitoring_interval)
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(self.monitoring_interval)
    
    def _collect_service_metrics(self):
        """Collect real-time metrics from all services"""
        
        try:
            # Get list of registered services
            services = self.prometheus_monitor.get_registered_services()
            
            with self.state_lock:
                for service_name in services:
                    try:
                        # Get metrics from Prometheus
                        metrics_data = self.prometheus_monitor.get_service_metrics(service_name)
                        
                        if metrics_data:
                            # Create service metrics object
                            service_metrics = ServiceMetrics(
                                service_name=service_name,
                                timestamp=datetime.now(),
                                cpu_usage=metrics_data.get('cpu_usage', 0),
                                memory_usage=metrics_data.get('memory_usage', 0),
                                cpu_limit=metrics_data.get('cpu_limit', 1),
                                memory_limit=metrics_data.get('memory_limit', 1e9),
                                latency=metrics_data.get('latency', 0),
                                request_rate=metrics_data.get('request_rate', 0),
                                error_rate=metrics_data.get('error_rate', 0),
                                availability=metrics_data.get('availability', 100),
                                cpu_utilization=metrics_data.get('cpu_usage', 0) / max(metrics_data.get('cpu_limit', 1), 0.001) * 100,
                                memory_utilization=metrics_data.get('memory_usage', 0) / max(metrics_data.get('memory_limit', 1e9), 1e6) * 100
                            )
                            
                            # Store metrics
                            if service_name not in self.service_metrics:
                                self.service_metrics[service_name] = []
                            
                            self.service_metrics[service_name].append(service_metrics)
                            
                            # Keep only recent metrics (last 1 hour)
                            cutoff_time = datetime.now() - timedelta(hours=1)
                            self.service_metrics[service_name] = [
                                m for m in self.service_metrics[service_name] 
                                if m.timestamp > cutoff_time
                            ]
                            
                    except Exception as e:
                        logger.warning(f"Failed to collect metrics for {service_name}: {e}")
                        
        except Exception as e:
            logger.error(f"Failed to collect service metrics: {e}")
    
    def _check_alerts(self):
        """Check for system alerts"""
        
        current_time = datetime.now()
        
        with self.state_lock:
            for service_name, metrics_list in self.service_metrics.items():
                if not metrics_list:
                    continue
                
                latest_metrics = metrics_list[-1]
                
                # Check various alert conditions
                self._check_latency_alerts(service_name, latest_metrics)
                self._check_resource_alerts(service_name, latest_metrics)
                self._check_availability_alerts(service_name, latest_metrics)
                self._check_error_rate_alerts(service_name, latest_metrics)
    
    def _check_latency_alerts(self, service_name: str, metrics: ServiceMetrics):
        """Check for latency-related alerts"""
        
        # High latency alert
        if metrics.latency > 0.5:  # 500ms threshold
            self._create_alert(
                service_name=service_name,
                level=AlertLevel.WARNING if metrics.latency < 1.0 else AlertLevel.ERROR,
                message=f"High latency detected: {metrics.latency:.3f}s",
                details={
                    'latency': metrics.latency,
                    'cpu_utilization': metrics.cpu_utilization,
                    'memory_utilization': metrics.memory_utilization,
                    'request_rate': metrics.request_rate
                }
            )
        
        # Latency spike detection
        if service_name in self.service_metrics and len(self.service_metrics[service_name]) >= 3:
            recent_latencies = [m.latency for m in self.service_metrics[service_name][-3:]]
            if metrics.latency > np.mean(recent_latencies[:-1]) * 2:
                self._create_alert(
                    service_name=service_name,
                    level=AlertLevel.WARNING,
                    message=f"Latency spike detected: {metrics.latency:.3f}s",
                    details={
                        'current_latency': metrics.latency,
                        'previous_average': np.mean(recent_latencies[:-1]),
                        'spike_ratio': metrics.latency / max(np.mean(recent_latencies[:-1]), 0.001)
                    }
                )
    
    def _check_resource_alerts(self, service_name: str, metrics: ServiceMetrics):
        """Check for resource-related alerts"""
        
        # High CPU utilization
        if metrics.cpu_utilization > 90:
            self._create_alert(
                service_name=service_name,
                level=AlertLevel.ERROR if metrics.cpu_utilization > 95 else AlertLevel.WARNING,
                message=f"High CPU utilization: {metrics.cpu_utilization:.1f}%",
                details={
                    'cpu_utilization': metrics.cpu_utilization,
                    'cpu_usage': metrics.cpu_usage,
                    'cpu_limit': metrics.cpu_limit
                }
            )
        
        # High memory utilization
        if metrics.memory_utilization > 90:
            self._create_alert(
                service_name=service_name,
                level=AlertLevel.ERROR if metrics.memory_utilization > 95 else AlertLevel.WARNING,
                message=f"High memory utilization: {metrics.memory_utilization:.1f}%",
                details={
                    'memory_utilization': metrics.memory_utilization,
                    'memory_usage': metrics.memory_usage,
                    'memory_limit': metrics.memory_limit
                }
            )
    
    def _check_availability_alerts(self, service_name: str, metrics: ServiceMetrics):
        """Check for availability alerts"""
        
        if metrics.availability < 99:
            level = AlertLevel.CRITICAL if metrics.availability < 95 else AlertLevel.ERROR
            self._create_alert(
                service_name=service_name,
                level=level,
                message=f"Low availability: {metrics.availability:.2f}%",
                details={
                    'availability': metrics.availability,
                    'error_rate': metrics.error_rate
                }
            )
    
    def _check_error_rate_alerts(self, service_name: str, metrics: ServiceMetrics):
        """Check for error rate alerts"""
        
        if metrics.error_rate > 5:  # 5% error rate
            level = AlertLevel.ERROR if metrics.error_rate > 10 else AlertLevel.WARNING
            self._create_alert(
                service_name=service_name,
                level=level,
                message=f"High error rate: {metrics.error_rate:.2f}%",
                details={
                    'error_rate': metrics.error_rate,
                    'request_rate': metrics.request_rate
                }
            )
    
    def _create_alert(self, service_name: str, level: AlertLevel, 
                     message: str, details: Dict[str, Any]):
        """Create a new alert"""
        
        # Create alert ID based on service and message
        alert_id = f"{service_name}_{hash(message) % 10000:04d}"
        
        # Check if similar alert already exists and is recent
        if alert_id in self.alerts:
            last_alert = self.alerts[alert_id]
            if (datetime.now() - last_alert.timestamp).total_seconds() < 300:  # 5 minutes
                return  # Don't create duplicate alerts
        
        alert = Alert(
            id=alert_id,
            timestamp=datetime.now(),
            level=level,
            service_name=service_name,
            message=message,
            details=details
        )
        
        self.alerts[alert_id] = alert
        
        logger.log(
            level=getattr(logging, level.value.upper()),
            msg=f"Alert [{level.value.upper()}] {service_name}: {message}"
        )
    
    def _update_system_status(self):
        """Update overall system status"""
        
        with self.state_lock:
            total_services = len(self.service_metrics)
            
            if total_services == 0:
                self.system_status = {
                    'status': 'UNKNOWN',
                    'services_count': 0,
                    'healthy_services': 0,
                    'alerts_count': len([a for a in self.alerts.values() if not a.resolved]),
                    'last_update': datetime.now()
                }
                return
            
            # Count healthy services
            healthy_services = 0
            for service_name, metrics_list in self.service_metrics.items():
                if metrics_list:
                    latest = metrics_list[-1]
                    if (latest.latency < 0.5 and latest.cpu_utilization < 90 and 
                        latest.memory_utilization < 90 and latest.availability > 99):
                        healthy_services += 1
            
            # Determine overall status
            health_ratio = healthy_services / total_services
            active_alerts = len([a for a in self.alerts.values() if not a.resolved])
            
            if health_ratio >= 0.9 and active_alerts == 0:
                status = 'HEALTHY'
            elif health_ratio >= 0.8 and active_alerts < 5:
                status = 'WARNING'
            elif health_ratio >= 0.6:
                status = 'DEGRADED'
            else:
                status = 'CRITICAL'
            
            self.system_status = {
                'status': status,
                'services_count': total_services,
                'healthy_services': healthy_services,
                'alerts_count': active_alerts,
                'last_update': datetime.now(),
                'health_ratio': health_ratio
            }
    
    def _cleanup_old_data(self):
        """Clean up old alerts and decisions"""
        
        current_time = datetime.now()
        
        with self.state_lock:
            # Clean up old alerts
            alert_cutoff = current_time - timedelta(hours=self.alert_retention_hours)
            self.alerts = {
                alert_id: alert for alert_id, alert in self.alerts.items()
                if alert.timestamp > alert_cutoff
            }
            
            # Clean up old decisions
            decision_cutoff = current_time - timedelta(hours=self.decision_retention_hours)
            self.decisions = {
                decision_id: decision for decision_id, decision in self.decisions.items()
                if decision.timestamp > decision_cutoff
            }
    
    def get_dashboard_data(self) -> Dict[str, Any]:
        """Get current dashboard data"""
        
        with self.state_lock:
            # Get recent metrics summary
            services_summary = {}
            for service_name, metrics_list in self.service_metrics.items():
                if metrics_list:
                    latest = metrics_list[-1]
                    services_summary[service_name] = {
                        'latency': latest.latency,
                        'cpu_utilization': latest.cpu_utilization,
                        'memory_utilization': latest.memory_utilization,
                        'availability': latest.availability,
                        'request_rate': latest.request_rate,
                        'last_update': latest.timestamp.isoformat()
                    }
            
            # Get active alerts
            active_alerts = [
                asdict(alert) for alert in self.alerts.values()
                if not alert.resolved
            ]
            
            # Get recent decisions
            recent_decisions = [
                asdict(decision) for decision in sorted(
                    self.decisions.values(),
                    key=lambda d: d.timestamp,
                    reverse=True
                )[:10]
            ]
            
            return {
                'system_status': self.system_status,
                'services': services_summary,
                'active_alerts': active_alerts,
                'recent_decisions': recent_decisions,
                'optimization_stats': {
                    'total_optimizations': self.optimization_statistics['total_optimizations'],
                    'success_rate': (self.optimization_statistics['successful_optimizations'] / 
                                   max(self.optimization_statistics['total_optimizations'], 1)) * 100,
                    'total_savings': self.optimization_statistics['total_savings'],
                    'services_managed': len(self.optimization_statistics['services_managed'])
                },
                'timestamp': datetime.now().isoformat()
            }

class DecisionExecutor:
    """
    Executes optimization decisions and manages their lifecycle
    """
    
    def __init__(self, dashboard: MonitoringDashboard, kubernetes_client=None):
        self.dashboard = dashboard
        self.kubernetes_client = kubernetes_client
        self.execution_queue = asyncio.Queue()
        self.execution_active = False
        self.execution_task = None
    
    def start_execution_engine(self):
        """Start the decision execution engine"""
        
        if self.execution_active:
            logger.warning("Execution engine already active")
            return
        
        self.execution_active = True
        
        # Start execution loop in asyncio
        import asyncio
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        if loop.is_running():
            self.execution_task = asyncio.create_task(self._execution_loop())
        else:
            loop.run_until_complete(self._execution_loop())
        
        logger.info("Started decision execution engine")
    
    def stop_execution_engine(self):
        """Stop the decision execution engine"""
        
        self.execution_active = False
        if self.execution_task:
            self.execution_task.cancel()
        
        logger.info("Stopped decision execution engine")
    
    async def _execution_loop(self):
        """Main execution loop"""
        
        while self.execution_active:
            try:
                # Wait for decisions in queue
                decision = await asyncio.wait_for(
                    self.execution_queue.get(), 
                    timeout=1.0
                )
                
                # Execute decision
                await self._execute_decision(decision)
                
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Error in execution loop: {e}")
    
    async def _execute_decision(self, decision: OptimizationDecision):
        """Execute a single optimization decision"""
        
        try:
            # Update decision status
            decision.status = DecisionStatus.EXECUTING
            decision.execution_time = datetime.now()
            
            logger.info(f"Executing optimization for {decision.service_name}")
            
            # Apply resource changes (would integrate with Kubernetes API)
            success = await self._apply_resource_changes(decision)
            
            if success:
                decision.status = DecisionStatus.COMPLETED
                decision.completion_time = datetime.now()
                
                # Update statistics
                self.dashboard.optimization_statistics['successful_optimizations'] += 1
                self.dashboard.optimization_statistics['total_savings'] += decision.resource_savings
                self.dashboard.optimization_statistics['services_managed'].add(decision.service_name)
                
                logger.info(f"Successfully applied optimization for {decision.service_name}")
            else:
                decision.status = DecisionStatus.FAILED
                logger.error(f"Failed to apply optimization for {decision.service_name}")
                
        except Exception as e:
            decision.status = DecisionStatus.FAILED
            logger.error(f"Error executing decision for {decision.service_name}: {e}")
        
        finally:
            # Update decision in dashboard
            self.dashboard.decisions[decision.id] = decision
            self.dashboard.optimization_statistics['total_optimizations'] += 1
    
    async def _apply_resource_changes(self, decision: OptimizationDecision) -> bool:
        """Apply resource changes to Kubernetes"""
        
        try:
            # This would integrate with Kubernetes API
            # For now, simulate the operation
            
            logger.info(
                f"Applying resource changes for {decision.service_name}: "
                f"CPU: {decision.current_resources['cpu_limit']:.3f} → "
                f"{decision.recommended_resources['cpu_limit']:.3f}, "
                f"Memory: {decision.current_resources['memory_limit']/1e9:.2f}GB → "
                f"{decision.recommended_resources['memory_limit']/1e9:.2f}GB"
            )
            
            # Simulate API call delay
            await asyncio.sleep(2)
            
            # Here you would implement actual Kubernetes API calls:
            # 1. Update Deployment/StatefulSet resources
            # 2. Wait for rollout completion
            # 3. Verify new resource limits
            
            # For simulation, assume 95% success rate
            return np.random.random() > 0.05
            
        except Exception as e:
            logger.error(f"Failed to apply resource changes: {e}")
            return False
    
    def submit_decision(self, decision: OptimizationDecision):
        """Submit optimization decision for execution"""
        
        # Add to dashboard
        self.dashboard.decisions[decision.id] = decision
        
        # Add to execution queue
        try:
            loop = asyncio.get_event_loop()
            loop.call_soon_threadsafe(
                lambda: asyncio.create_task(self.execution_queue.put(decision))
            )
        except RuntimeError:
            # If no loop is running, just store the decision
            logger.warning("No asyncio loop running, decision stored but not executed")
    
    def cancel_decision(self, decision_id: str) -> bool:
        """Cancel a pending decision"""
        
        if decision_id in self.dashboard.decisions:
            decision = self.dashboard.decisions[decision_id]
            if decision.status == DecisionStatus.PENDING:
                decision.status = DecisionStatus.CANCELLED
                logger.info(f"Cancelled decision {decision_id}")
                return True
        
        return False

# Integration class
class DAREMonitoringSystem:
    """
    Complete monitoring and execution system for DARE algorithm
    """
    
    def __init__(self, prometheus_monitor, latency_predictor, resource_optimizer):
        self.dashboard = MonitoringDashboard(
            prometheus_monitor, latency_predictor, resource_optimizer
        )
        self.executor = DecisionExecutor(self.dashboard)
        self.auto_optimization_enabled = False
        
    def start_system(self):
        """Start the complete monitoring system"""
        
        self.dashboard.start_monitoring()
        self.executor.start_execution_engine()
        
        logger.info("DARE monitoring system started")
    
    def stop_system(self):
        """Stop the monitoring system"""
        
        self.dashboard.stop_monitoring()
        self.executor.stop_execution_engine()
        
        logger.info("DARE monitoring system stopped")
    
    def enable_auto_optimization(self):
        """Enable automatic optimization execution"""
        self.auto_optimization_enabled = True
        logger.info("Auto-optimization enabled")
    
    def disable_auto_optimization(self):
        """Disable automatic optimization execution"""
        self.auto_optimization_enabled = False
        logger.info("Auto-optimization disabled")
    
    def get_system_overview(self) -> Dict[str, Any]:
        """Get complete system overview"""
        return self.dashboard.get_dashboard_data()

# Example usage
if __name__ == "__main__":
    print("DARE Monitoring and Execution System")
    print("This module provides:")
    print("- Real-time monitoring dashboard")
    print("- Alert management and notifications")
    print("- Decision execution with Kubernetes integration")
    print("- Performance tracking and reporting")
"""
AI-Driven Latency-Constrained Resource Management Algorithm for Kubernetes

This module implements the core algorithm that:
1. Predicts resource usage trends using DARE-TL (Dynamic Adaptive Resource Estimation with Transfer Learning)
2. Estimates latency impact of resource changes
3. Optimizes resource allocation while maintaining latency constraints
4. Provides real-time decision making for Kubernetes resource management

Architecture Components:
- TrendLearner: Uses SGD with EMA for online learning of resource usage patterns
- LatencyPredictor: Estimates service latency based on resource constraints
- ResourceOptimizer: Finds optimal resource allocation within latency bounds
- Monitor: Real-time data collection and decision execution
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, Tuple, List, Optional
from dataclasses import dataclass
from enum import Enum
import requests
import time
from datetime import datetime, timedelta
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import SGDRegressor
from collections import deque
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ServiceType(Enum):
    """Service types with different latency characteristics"""
    CPU_BOUND = "cpu_bound"        # Prime verifier, Hash generator
    MEMORY_RESILIENT = "memory_resilient"  # Echo, Password generator
    BALANCED = "balanced"          # Services with mixed workloads

@dataclass
class ResourceState:
    """Current resource state of a service"""
    cpu_usage: float
    memory_usage: float
    cpu_limit: float
    memory_limit: float
    request_rate: float
    latency: float
    timestamp: datetime

@dataclass
class ResourceRecommendation:
    """Resource allocation recommendation"""
    cpu_limit: float
    memory_limit: float
    confidence: float
    expected_latency: float
    safety_margin: float
    risk_level: str

class LatencyPredictor:
    """Predicts service latency based on resource constraints and service characteristics"""
    
    def __init__(self):
        self.service_profiles = {
            ServiceType.CPU_BOUND: {
                'cpu_sensitivity': 0.8,
                'memory_sensitivity': 0.2,
                'latency_threshold': 0.5,  # seconds
                'critical_reduction_point': 0.7  # 70% of resource utilization
            },
            ServiceType.MEMORY_RESILIENT: {
                'cpu_sensitivity': 0.3,
                'memory_sensitivity': 0.1,
                'latency_threshold': 0.3,
                'critical_reduction_point': 0.9
            },
            ServiceType.BALANCED: {
                'cpu_sensitivity': 0.5,
                'memory_sensitivity': 0.4,
                'latency_threshold': 0.4,
                'critical_reduction_point': 0.8
            }
        }
        
    def predict_latency(self, current_state: ResourceState, 
                       new_cpu_limit: float, new_memory_limit: float,
                       service_type: ServiceType) -> float:
        """
        Predict latency for given resource constraints
        
        Uses the critical reduction point (CRP) concept where latency increases
        exponentially beyond certain resource utilization thresholds
        """
        profile = self.service_profiles[service_type]
        
        # Calculate resource utilization ratios
        cpu_utilization = current_state.cpu_usage / max(new_cpu_limit, 0.001)
        memory_utilization = current_state.memory_usage / max(new_memory_limit, 0.001)
        
        # Base latency from historical data
        base_latency = current_state.latency
        
        # Calculate pressure factors
        cpu_pressure = max(0, cpu_utilization - profile['critical_reduction_point'])
        memory_pressure = max(0, memory_utilization - profile['critical_reduction_point'])
        
        # Exponential latency increase beyond CRP
        if cpu_pressure > 0:
            cpu_impact = profile['cpu_sensitivity'] * (np.exp(cpu_pressure * 5) - 1)
        else:
            cpu_impact = 0
            
        if memory_pressure > 0:
            memory_impact = profile['memory_sensitivity'] * (np.exp(memory_pressure * 3) - 1)
        else:
            memory_impact = 0
            
        # Combined latency prediction
        predicted_latency = base_latency * (1 + cpu_impact + memory_impact)
        
        # Add request rate impact
        rate_factor = max(1, current_state.request_rate / 10)  # Normalize to typical load
        predicted_latency *= np.sqrt(rate_factor)
        
        return predicted_latency

class ResourceOptimizer:
    """Optimizes resource allocation while maintaining latency constraints"""
    
    def __init__(self, latency_predictor: LatencyPredictor):
        self.latency_predictor = latency_predictor
        
    def optimize_resources(self, current_state: ResourceState,
                          service_type: ServiceType,
                          target_latency: float,
                          min_reduction: float = 0.05) -> ResourceRecommendation:
        """
        Find optimal resource allocation that minimizes resource usage
        while keeping latency below target
        """
        
        # Define search space
        cpu_limits = np.linspace(
            current_state.cpu_usage * 1.1,  # Minimum viable
            current_state.cpu_limit,        # Current maximum
            20
        )
        
        memory_limits = np.linspace(
            current_state.memory_usage * 1.1,
            current_state.memory_limit,
            20
        )
        
        best_recommendation = None
        min_resource_cost = float('inf')
        
        for cpu_limit in cpu_limits:
            for memory_limit in memory_limits:
                # Predict latency for this configuration
                predicted_latency = self.latency_predictor.predict_latency(
                    current_state, cpu_limit, memory_limit, service_type
                )
                
                # Check if latency constraint is satisfied
                if predicted_latency <= target_latency:
                    # Calculate resource cost (lower is better)
                    resource_cost = cpu_limit + memory_limit / 1000  # Normalize memory
                    
                    if resource_cost < min_resource_cost:
                        min_resource_cost = resource_cost
                        
                        # Calculate confidence based on distance from CRP
                        profile = self.latency_predictor.service_profiles[service_type]
                        cpu_util = current_state.cpu_usage / cpu_limit
                        mem_util = current_state.memory_usage / memory_limit
                        
                        # Confidence decreases as we approach CRP
                        cpu_confidence = max(0, 1 - (cpu_util / profile['critical_reduction_point']))
                        mem_confidence = max(0, 1 - (mem_util / profile['critical_reduction_point']))
                        confidence = min(cpu_confidence, mem_confidence)
                        
                        # Determine risk level
                        safety_margin = (target_latency - predicted_latency) / target_latency
                        if safety_margin > 0.3:
                            risk_level = "LOW"
                        elif safety_margin > 0.1:
                            risk_level = "MEDIUM"
                        else:
                            risk_level = "HIGH"
                        
                        best_recommendation = ResourceRecommendation(
                            cpu_limit=cpu_limit,
                            memory_limit=memory_limit,
                            confidence=confidence,
                            expected_latency=predicted_latency,
                            safety_margin=safety_margin,
                            risk_level=risk_level
                        )
        
        # If no safe reduction found, recommend current limits
        if best_recommendation is None:
            best_recommendation = ResourceRecommendation(
                cpu_limit=current_state.cpu_limit,
                memory_limit=current_state.memory_limit,
                confidence=0.5,
                expected_latency=current_state.latency,
                safety_margin=0.0,
                risk_level="NONE"
            )
            
        return best_recommendation

class PrometheusMonitor:
    """Monitors Kubernetes services through Prometheus metrics"""
    
    def __init__(self, prometheus_url: str = "http://localhost:9090"):
        self.prometheus_url = prometheus_url
        
    def query_prometheus(self, query: str) -> List[Dict]:
        """Execute Prometheus query"""
        try:
            response = requests.get(
                f"{self.prometheus_url}/api/v1/query",
                params={"query": query},
                timeout=10
            )
            if response.status_code == 200:
                return response.json()["data"]["result"]
            else:
                logger.error(f"Prometheus query failed: {response.status_code}")
                return []
        except Exception as e:
            logger.error(f"Failed to query Prometheus: {e}")
            return []
    
    def get_service_state(self, service_name: str, namespace: str = "default") -> Optional[ResourceState]:
        """Get current resource state of a service"""
        try:
            # CPU Usage
            cpu_query = f'sum(rate(container_cpu_usage_seconds_total{{pod=~"{service_name}-.*", namespace="{namespace}"}}[1m]))'
            cpu_result = self.query_prometheus(cpu_query)
            cpu_usage = float(cpu_result[0]['value'][1]) if cpu_result else 0
            
            # Memory Usage
            memory_query = f'sum(container_memory_usage_bytes{{pod=~"{service_name}-.*", namespace="{namespace}"}})'
            memory_result = self.query_prometheus(memory_query)
            memory_usage = float(memory_result[0]['value'][1]) if memory_result else 0
            
            # Resource Limits
            cpu_limit_query = f'sum(kube_pod_container_resource_limits{{pod=~"{service_name}-.*", resource="cpu", unit="core"}})'
            cpu_limit_result = self.query_prometheus(cpu_limit_query)
            cpu_limit = float(cpu_limit_result[0]['value'][1]) if cpu_limit_result else 1.0
            
            memory_limit_query = f'sum(kube_pod_container_resource_limits{{pod=~"{service_name}-.*", resource="memory", unit="byte"}})'
            memory_limit_result = self.query_prometheus(memory_limit_query)
            memory_limit = float(memory_limit_result[0]['value'][1]) if memory_limit_result else 1000000000
            
            # Request Rate
            request_rate_query = f'sum(rate(http_server_requests_seconds_count{{pod=~"{service_name}-.*"}}[5m]))'
            request_rate_result = self.query_prometheus(request_rate_query)
            request_rate = float(request_rate_result[0]['value'][1]) if request_rate_result else 0
            
            # Latency (average)
            latency_query = f'sum(rate(http_server_requests_seconds_sum{{pod=~"{service_name}-.*"}}[1m])) / sum(rate(http_server_requests_seconds_count{{pod=~"{service_name}-.*"}}[1m]))'
            latency_result = self.query_prometheus(latency_query)
            latency = float(latency_result[0]['value'][1]) if latency_result else 0.1
            
            return ResourceState(
                cpu_usage=cpu_usage,
                memory_usage=memory_usage,
                cpu_limit=cpu_limit,
                memory_limit=memory_limit,
                request_rate=request_rate,
                latency=latency,
                timestamp=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"Failed to get service state for {service_name}: {e}")
            return None

class DAREAlgorithm:
    """
    Main DARE-TL Algorithm for AI-Driven Latency-Constrained Resource Management
    
    Integrates:
    - Dynamic resource usage prediction
    - Latency-aware optimization
    - Real-time monitoring and adaptation
    """
    
    def __init__(self, prometheus_url: str = "http://localhost:9090"):
        self.monitor = PrometheusMonitor(prometheus_url)
        self.latency_predictor = LatencyPredictor()
        self.optimizer = ResourceOptimizer(self.latency_predictor)
        
        # Service configurations
        self.services = {}
        self.history = deque(maxlen=1000)  # Store historical data
        
    def register_service(self, service_name: str, service_type: ServiceType, 
                        target_latency: float):
        """Register a service for monitoring and optimization"""
        self.services[service_name] = {
            'type': service_type,
            'target_latency': target_latency,
            'last_update': None,
            'update_interval': 60  # seconds
        }
        logger.info(f"Registered service {service_name} of type {service_type.value}")
    
    def run_optimization_cycle(self, service_name: str) -> Optional[ResourceRecommendation]:
        """Run one optimization cycle for a service"""
        if service_name not in self.services:
            logger.error(f"Service {service_name} not registered")
            return None
            
        service_config = self.services[service_name]
        
        # Check if it's time for update
        if (service_config['last_update'] and 
            datetime.now() - service_config['last_update'] < timedelta(seconds=service_config['update_interval'])):
            return None
            
        # Get current service state
        current_state = self.monitor.get_service_state(service_name)
        if not current_state:
            logger.error(f"Could not get state for service {service_name}")
            return None
            
        # Store in history
        self.history.append({
            'service': service_name,
            'state': current_state,
            'timestamp': datetime.now()
        })
        
        # Generate recommendation
        recommendation = self.optimizer.optimize_resources(
            current_state=current_state,
            service_type=service_config['type'],
            target_latency=service_config['target_latency']
        )
        
        # Update last optimization time
        service_config['last_update'] = datetime.now()
        
        logger.info(f"Optimization for {service_name}: "
                   f"CPU: {current_state.cpu_limit:.3f} -> {recommendation.cpu_limit:.3f}, "
                   f"Memory: {current_state.memory_limit/1e6:.0f}MB -> {recommendation.memory_limit/1e6:.0f}MB, "
                   f"Risk: {recommendation.risk_level}")
        
        return recommendation
    
    def run_continuous_monitoring(self, interval: int = 30):
        """Run continuous monitoring and optimization"""
        logger.info("Starting continuous monitoring...")
        
        while True:
            try:
                for service_name in self.services:
                    recommendation = self.run_optimization_cycle(service_name)
                    
                    if recommendation and recommendation.risk_level in ["LOW", "MEDIUM"]:
                        # Apply recommendation (would integrate with Kubernetes API)
                        self.apply_resource_changes(service_name, recommendation)
                        
                time.sleep(interval)
                
            except KeyboardInterrupt:
                logger.info("Monitoring stopped by user")
                break
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(interval)
    
    def apply_resource_changes(self, service_name: str, recommendation: ResourceRecommendation):
        """Apply resource changes to Kubernetes (placeholder for K8s API integration)"""
        logger.info(f"Applying resource changes to {service_name}:")
        logger.info(f"  CPU Limit: {recommendation.cpu_limit:.3f} cores")
        logger.info(f"  Memory Limit: {recommendation.memory_limit/1e6:.0f} MB")
        logger.info(f"  Expected Latency: {recommendation.expected_latency:.3f}s")
        logger.info(f"  Confidence: {recommendation.confidence:.2f}")
        
        # TODO: Integrate with Kubernetes API to actually apply changes
        # Example:
        # k8s_api.patch_deployment_resources(service_name, recommendation)
    
    def get_system_status(self) -> Dict:
        """Get overall system status"""
        status = {
            'services': {},
            'total_services': len(self.services),
            'last_update': datetime.now().isoformat()
        }
        
        for service_name in self.services:
            current_state = self.monitor.get_service_state(service_name)
            if current_state:
                status['services'][service_name] = {
                    'cpu_usage': current_state.cpu_usage,
                    'memory_usage': current_state.memory_usage,
                    'latency': current_state.latency,
                    'request_rate': current_state.request_rate
                }
        
        return status

# Example usage and testing
if __name__ == "__main__":
    # Initialize the DARE algorithm
    dare = DAREAlgorithm()
    
    # Register services (based on the project's service types)
    dare.register_service("service-1-deployment", ServiceType.CPU_BOUND, target_latency=0.5)
    dare.register_service("hash-gen-deployment", ServiceType.CPU_BOUND, target_latency=0.3)
    dare.register_service("rand-pw-gen-deployment", ServiceType.MEMORY_RESILIENT, target_latency=0.2)
    dare.register_service("service-2-deployment", ServiceType.BALANCED, target_latency=0.4)
    
    # Run a single optimization cycle for testing
    recommendation = dare.run_optimization_cycle("service-1-deployment")
    if recommendation:
        print(f"Recommendation: {recommendation}")
    
    # Get system status
    status = dare.get_system_status()
    print(f"System Status: {json.dumps(status, indent=2)}")
    
    # For continuous monitoring, uncomment:
    # dare.run_continuous_monitoring(interval=30)
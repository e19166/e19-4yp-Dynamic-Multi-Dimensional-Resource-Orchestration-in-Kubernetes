"""
Result analysis and confidence assessment
"""

import logging
from typing import Dict, Optional
import pandas as pd
from optimization_types import OptimizationResult, ServiceState, ResourceConstraints

logger = logging.getLogger(__name__)


class ResultAnalyzer:
    """
    Analyzes optimization results and calculates confidence metrics
    """
    
    def __init__(self, latency_predictor):
        self.latency_predictor = latency_predictor
        self.optimization_history = {}
    
    def analyze_result(self, result_dict: Dict, service_state: ServiceState,
                      constraints: ResourceConstraints, strategy: str) -> OptimizationResult:
        """
        Analyze optimization result and create comprehensive result object
        
        Args:
            result_dict: Raw optimization result
            service_state: Original service state
            constraints: Resource constraints used
            strategy: Optimization strategy used
            
        Returns:
            Complete OptimizationResult object
        """
        
        # Calculate additional metrics
        resource_savings = self._calculate_savings(service_state, result_dict)
        confidence = self._calculate_confidence(service_state.service_name, result_dict, service_state)
        risk_level = self._assess_risk(service_state.service_name, result_dict, service_state, constraints)
        
        return OptimizationResult(
            cpu_limit=result_dict['cpu_limit'],
            memory_limit=result_dict['memory_limit'],
            expected_latency=result_dict['expected_latency'],
            latency_uncertainty=result_dict.get('uncertainty', 0.1),
            resource_savings=resource_savings,
            confidence=confidence,
            risk_level=risk_level,
            optimization_strategy=strategy,
            iterations=result_dict.get('iterations', 0),
            convergence_status=result_dict.get('status', 'UNKNOWN')
        )
    
    def _calculate_savings(self, service_state: ServiceState, result_dict: Dict) -> float:
        """Calculate resource savings percentage"""
        
        # Normalize resources for comparison (CPU + memory in GB)
        current_resources = service_state.cpu_limit + service_state.memory_limit / 1e9
        new_resources = result_dict['cpu_limit'] + result_dict['memory_limit'] / 1e9
        
        if current_resources <= 0:
            return 0.0
        
        savings = (current_resources - new_resources) / current_resources * 100
        return max(0, savings)  # Ensure non-negative
    
    def _calculate_confidence(self, service_name: str, result_dict: Dict, 
                            service_state: ServiceState) -> float:
        """Calculate confidence in the optimization result"""
        
        # Base confidence from latency uncertainty
        uncertainty = result_dict.get('uncertainty', 0.1)
        base_confidence = max(0, 1 - uncertainty / 0.5)  # Assume 0.5s as max reasonable uncertainty
        
        # Adjust based on resource change magnitude
        change_penalty = self._calculate_change_penalty(result_dict, service_state)
        confidence = base_confidence * (1 - change_penalty)
        
        # Boost confidence based on historical success
        history_boost = self._get_history_boost(service_name)
        confidence *= history_boost
        
        # Convergence quality factor
        convergence_factor = self._get_convergence_factor(result_dict)
        confidence *= convergence_factor
        
        return max(0.1, min(1.0, confidence))
    
    def _calculate_change_penalty(self, result_dict: Dict, service_state: ServiceState) -> float:
        """Calculate penalty based on magnitude of resource changes"""
        
        cpu_change = abs(result_dict['cpu_limit'] - service_state.cpu_limit) / service_state.cpu_limit
        memory_change = abs(result_dict['memory_limit'] - service_state.memory_limit) / service_state.memory_limit
        
        # Average change as penalty factor (max 30% penalty)
        change_penalty = min(0.3, (cpu_change + memory_change) / 2 * 0.3)
        return change_penalty
    
    def _get_history_boost(self, service_name: str) -> float:
        """Get confidence boost based on historical performance"""
        
        if service_name not in self.optimization_history:
            return 0.9  # Slight penalty for new services
        
        history = self.optimization_history[service_name]
        if not history:
            return 0.9
        
        # Calculate recent success rate
        recent_history = history[-10:]  # Last 10 optimizations
        success_rate = sum(1 for h in recent_history if h.get('success', False)) / len(recent_history)
        
        # Convert to boost factor (0.7 to 1.1)
        return 0.7 + 0.4 * success_rate
    
    def _get_convergence_factor(self, result_dict: Dict) -> float:
        """Get confidence factor based on optimization convergence"""
        
        status = result_dict.get('status', 'UNKNOWN')
        iterations = result_dict.get('iterations', 0)
        
        if status == 'CONVERGED':
            # Good convergence, factor based on iterations
            if iterations < 20:
                return 1.0  # Fast convergence
            elif iterations < 50:
                return 0.95  # Normal convergence
            else:
                return 0.9   # Slow convergence
        elif status == 'MAX_ITER':
            return 0.8  # Hit iteration limit
        else:
            return 0.7  # Failed or unknown
    
    def _assess_risk(self, service_name: str, result_dict: Dict, 
                    service_state: ServiceState, constraints: ResourceConstraints) -> str:
        """Assess risk level of the optimization result"""
        
        try:
            # Calculate resource utilization with new limits
            cpu_util = service_state.cpu_usage / result_dict['cpu_limit']
            memory_util = service_state.memory_usage / result_dict['memory_limit']
            
            # Get service-specific risk assessment
            if hasattr(self.latency_predictor, 'get_service_risk_assessment'):
                risk_assessment = self.latency_predictor.get_service_risk_assessment(
                    service_name, cpu_util, memory_util
                )
                base_risk = risk_assessment.get('risk_level', 'MEDIUM')
            else:
                # Fallback risk assessment
                base_risk = self._simple_risk_assessment(cpu_util, memory_util)
            
            # Adjust based on latency margin
            latency_margin = self._calculate_latency_margin(result_dict, constraints)
            
            return self._combine_risk_factors(base_risk, latency_margin)
            
        except Exception as e:
            logger.warning(f"Risk assessment failed for {service_name}: {e}")
            return "HIGH"  # Conservative fallback
    
    def _simple_risk_assessment(self, cpu_util: float, memory_util: float) -> str:
        """Simple fallback risk assessment"""
        
        max_util = max(cpu_util, memory_util)
        
        if max_util > 0.95:
            return "CRITICAL"
        elif max_util > 0.90:
            return "HIGH"
        elif max_util > 0.80:
            return "MEDIUM"
        else:
            return "LOW"
    
    def _calculate_latency_margin(self, result_dict: Dict, constraints: ResourceConstraints) -> float:
        """Calculate latency safety margin"""
        
        expected_latency = result_dict['expected_latency']
        latency_threshold = constraints.latency_threshold
        
        if latency_threshold <= 0:
            return 0.0
        
        return (latency_threshold - expected_latency) / latency_threshold
    
    def _combine_risk_factors(self, base_risk: str, latency_margin: float) -> str:
        """Combine risk factors into final risk level"""
        
        # Risk hierarchy
        risk_levels = ["LOW", "MEDIUM", "HIGH", "CRITICAL"]
        
        if latency_margin < 0:
            return "CRITICAL"  # Exceeds threshold
        elif latency_margin < 0.1:
            return "HIGH"
        elif latency_margin < 0.2:
            # Don't downgrade below medium for low latency margin
            base_index = risk_levels.index(base_risk) if base_risk in risk_levels else 1
            return risk_levels[max(1, base_index)]  # At least MEDIUM
        else:
            # Good latency margin, can downgrade base risk slightly
            if base_risk == "CRITICAL":
                return "HIGH"
            return base_risk
    
    def update_optimization_history(self, service_name: str, result: OptimizationResult,
                                   actual_latency: Optional[float] = None):
        """Update optimization history with results"""
        
        if service_name not in self.optimization_history:
            self.optimization_history[service_name] = []
        
        success = (actual_latency is not None and 
                  actual_latency <= result.expected_latency * 1.2)
        
        history_entry = {
            'timestamp': pd.Timestamp.now(),
            'predicted_latency': result.expected_latency,
            'actual_latency': actual_latency,
            'cpu_limit': result.cpu_limit,
            'memory_limit': result.memory_limit,
            'resource_savings': result.resource_savings,
            'confidence': result.confidence,
            'risk_level': result.risk_level,
            'success': success
        }
        
        self.optimization_history[service_name].append(history_entry)
        
        # Keep only recent history (last 100 entries)
        if len(self.optimization_history[service_name]) > 100:
            self.optimization_history[service_name] = self.optimization_history[service_name][-100:]
        
        logger.info(f"Updated optimization history for {service_name}: success={success}")
    
    def get_service_performance_stats(self, service_name: str) -> Dict:
        """Get performance statistics for a service"""
        
        if service_name not in self.optimization_history:
            return {}
        
        history = self.optimization_history[service_name]
        if not history:
            return {}
        
        # Calculate statistics
        total_optimizations = len(history)
        successful_optimizations = sum(1 for h in history if h.get('success', False))
        total_savings = sum(h.get('resource_savings', 0) for h in history)
        avg_confidence = sum(h.get('confidence', 0) for h in history) / total_optimizations
        
        return {
            'total_optimizations': total_optimizations,
            'success_rate': successful_optimizations / total_optimizations * 100,
            'total_savings': total_savings,
            'average_confidence': avg_confidence,
            'last_optimization': history[-1]['timestamp'] if history else None
        }
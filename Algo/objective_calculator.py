"""
Objective function calculator for multi-objective optimization
"""

import logging
from typing import Dict, Tuple
from optimization_types import ResourceConstraints, OptimizationWeights, ServiceState

logger = logging.getLogger(__name__)


class ObjectiveCalculator:
    """
    Calculates multi-objective function values for optimization
    """
    
    def __init__(self, latency_predictor):
        self.latency_predictor = latency_predictor
    
    def calculate_objective(self, resource_config: Tuple[float, float], 
                          service_state: ServiceState,
                          constraints: ResourceConstraints,
                          weights: OptimizationWeights) -> float:
        """
        Calculate multi-objective function value
        
        Args:
            resource_config: (cpu_limit, memory_limit)
            service_state: Current service state
            constraints: Resource constraints
            weights: Objective weights
            
        Returns:
            Total objective cost (lower is better)
        """
        
        cpu_limit, memory_limit = resource_config
        
        # Get latency prediction
        predicted_latency, uncertainty, _ = self.latency_predictor.predict_latency(
            service_name=service_state.service_name,
            cpu_usage=service_state.cpu_usage,
            memory_usage=service_state.memory_usage,
            cpu_limit=cpu_limit,
            memory_limit=memory_limit,
            request_rate=service_state.request_rate
        )
        
        # Calculate individual objectives
        resource_cost = self._calculate_resource_cost(
            resource_config, service_state
        )
        latency_penalty = self._calculate_latency_penalty(
            predicted_latency, constraints.latency_threshold
        )
        uncertainty_penalty = self._calculate_uncertainty_penalty(
            uncertainty, constraints.latency_threshold
        )
        stability_penalty = self._calculate_stability_penalty(
            resource_config, service_state
        )
        
        # Combine objectives with weights
        total_cost = (weights.resource * resource_cost +
                     weights.latency * latency_penalty +
                     weights.uncertainty * uncertainty_penalty +
                     weights.stability * stability_penalty)
        
        return total_cost
    
    def _calculate_resource_cost(self, resource_config: Tuple[float, float], 
                               service_state: ServiceState) -> float:
        """Calculate normalized resource cost"""
        
        cpu_limit, memory_limit = resource_config
        
        # Normalize by current resources
        cpu_cost = cpu_limit / service_state.cpu_limit
        memory_cost = memory_limit / service_state.memory_limit
        
        # Average cost (higher resources = higher cost)
        return (cpu_cost + memory_cost) / 2
    
    def _calculate_latency_penalty(self, predicted_latency: float, 
                                 latency_threshold: float) -> float:
        """Calculate latency constraint penalty"""
        
        # Penalty for exceeding threshold
        latency_violation = max(0, predicted_latency - latency_threshold)
        return latency_violation / latency_threshold
    
    def _calculate_uncertainty_penalty(self, uncertainty: float,
                                     latency_threshold: float) -> float:
        """Calculate uncertainty penalty (prefer confident predictions)"""
        
        return uncertainty / latency_threshold
    
    def _calculate_stability_penalty(self, resource_config: Tuple[float, float],
                                   service_state: ServiceState) -> float:
        """Calculate stability penalty (penalize large changes)"""
        
        cpu_limit, memory_limit = resource_config
        
        # Calculate relative changes
        cpu_change = abs(cpu_limit - service_state.cpu_limit) / service_state.cpu_limit
        memory_change = abs(memory_limit - service_state.memory_limit) / service_state.memory_limit
        
        # Average change penalty
        return (cpu_change + memory_change) / 2
    
    def calculate_latency_constraint(self, resource_config: Tuple[float, float],
                                   service_state: ServiceState,
                                   max_latency: float) -> float:
        """
        Calculate latency constraint for optimization
        
        Returns:
            Positive value if constraint satisfied, negative if violated
        """
        
        cpu_limit, memory_limit = resource_config
        
        predicted_latency, _, _ = self.latency_predictor.predict_latency(
            service_name=service_state.service_name,
            cpu_usage=service_state.cpu_usage,
            memory_usage=service_state.memory_usage,
            cpu_limit=cpu_limit,
            memory_limit=memory_limit,
            request_rate=service_state.request_rate
        )
        
        # Return margin (positive = good, negative = violation)
        return max_latency - predicted_latency
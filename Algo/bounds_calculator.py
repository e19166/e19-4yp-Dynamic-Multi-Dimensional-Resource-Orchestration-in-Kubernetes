"""
Bounds calculation for optimization constraints
"""

import logging
from typing import List, Tuple
from optimization_types import ResourceConstraints, ServiceState

logger = logging.getLogger(__name__)


class BoundsCalculator:
    """
    Calculates optimization bounds based on constraints and current state
    """
    
    def __init__(self):
        # No initialization needed for this calculator
        pass
    
    def calculate_bounds(self, service_state: ServiceState, 
                        constraints: ResourceConstraints) -> List[Tuple[float, float]]:
        """
        Calculate optimization bounds for CPU and memory
        
        Args:
            service_state: Current service state
            constraints: Resource constraints
            
        Returns:
            List of (min, max) tuples for [cpu_limit, memory_limit]
        """
        
        cpu_bounds = self._calculate_cpu_bounds(service_state, constraints)
        memory_bounds = self._calculate_memory_bounds(service_state, constraints)
        
        return [cpu_bounds, memory_bounds]
    
    def _calculate_cpu_bounds(self, service_state: ServiceState,
                            constraints: ResourceConstraints) -> Tuple[float, float]:
        """Calculate CPU limit bounds"""
        
        # Minimum CPU: max of usage-based minimum and absolute minimum
        min_cpu = max(
            service_state.cpu_usage * constraints.min_cpu_ratio,
            service_state.cpu_limit * 0.1  # Never go below 10% of current
        )
        
        # Maximum CPU: ratio of current limit
        max_cpu = service_state.cpu_limit * constraints.max_cpu_ratio
        
        # Ensure minimum doesn't exceed maximum
        if min_cpu > max_cpu:
            logger.warning(f"CPU min bound ({min_cpu}) exceeds max bound ({max_cpu}), adjusting")
            min_cpu = max_cpu * 0.8
        
        return (min_cpu, max_cpu)
    
    def _calculate_memory_bounds(self, service_state: ServiceState,
                               constraints: ResourceConstraints) -> Tuple[float, float]:
        """Calculate memory limit bounds"""
        
        # Minimum memory: max of usage-based minimum and absolute minimum
        min_memory = max(
            service_state.memory_usage * constraints.min_memory_ratio,
            service_state.memory_limit * 0.1  # Never go below 10% of current
        )
        
        # Maximum memory: ratio of current limit
        max_memory = service_state.memory_limit * constraints.max_memory_ratio
        
        # Ensure minimum doesn't exceed maximum
        if min_memory > max_memory:
            logger.warning(f"Memory min bound ({min_memory}) exceeds max bound ({max_memory}), adjusting")
            min_memory = max_memory * 0.8
        
        return (min_memory, max_memory)
    
    def validate_bounds(self, bounds: List[Tuple[float, float]], 
                       service_state: ServiceState) -> bool:
        """
        Validate that bounds are reasonable
        
        Args:
            bounds: List of (min, max) bounds
            service_state: Current service state
            
        Returns:
            True if bounds are valid
        """
        
        if len(bounds) != 2:
            return False
        
        cpu_bounds, memory_bounds = bounds
        
        # Check CPU bounds
        if cpu_bounds[0] <= 0 or cpu_bounds[1] <= cpu_bounds[0]:
            logger.error(f"Invalid CPU bounds: {cpu_bounds}")
            return False
        
        # Check memory bounds
        if memory_bounds[0] <= 0 or memory_bounds[1] <= memory_bounds[0]:
            logger.error(f"Invalid memory bounds: {memory_bounds}")
            return False
        
        # Check that bounds don't allow extremely small allocations
        min_cpu_ratio = cpu_bounds[0] / service_state.cpu_usage
        min_memory_ratio = memory_bounds[0] / service_state.memory_usage
        
        if min_cpu_ratio < 1.01 or min_memory_ratio < 1.01:
            logger.warning("Bounds allow very tight resource allocation")
        
        return True
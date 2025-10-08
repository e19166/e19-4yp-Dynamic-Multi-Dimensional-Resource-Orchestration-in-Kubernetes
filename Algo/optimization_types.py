"""
Core types and enums for resource optimization
"""

from enum import Enum
from dataclasses import dataclass
from typing import Dict, Any, Optional
from datetime import datetime


class OptimizationStrategy(Enum):
    """Different optimization strategies for different scenarios"""
    CONSERVATIVE = "conservative"    # Prioritize stability and safety
    AGGRESSIVE = "aggressive"       # Maximize resource reduction
    BALANCED = "balanced"           # Balance between safety and efficiency
    ADAPTIVE = "adaptive"           # Adapt strategy based on service behavior


@dataclass
class ResourceConstraints:
    """Resource constraints for optimization"""
    min_cpu_ratio: float = 1.1      # Minimum CPU as ratio of current usage
    max_cpu_ratio: float = 2.0      # Maximum CPU as ratio of current usage
    min_memory_ratio: float = 1.1   # Minimum memory as ratio of current usage
    max_memory_ratio: float = 2.0   # Maximum memory as ratio of current usage
    latency_threshold: float = 0.5  # Maximum acceptable latency
    safety_margin: float = 0.15     # Safety margin for resource allocation


@dataclass
class OptimizationResult:
    """Result of resource optimization"""
    cpu_limit: float
    memory_limit: float
    expected_latency: float
    latency_uncertainty: float
    resource_savings: float         # Percentage of resource saved
    confidence: float              # Confidence in the recommendation
    risk_level: str               # LOW, MEDIUM, HIGH, CRITICAL
    optimization_strategy: str    # Strategy used
    iterations: int               # Number of optimization iterations
    convergence_status: str       # CONVERGED, MAX_ITER, FAILED


@dataclass
class ServiceState:
    """Current state of a service"""
    service_name: str
    cpu_usage: float
    memory_usage: float
    cpu_limit: float
    memory_limit: float
    request_rate: float
    latency: Optional[float] = None
    timestamp: Optional[datetime] = None


@dataclass
class OptimizationWeights:
    """Weights for different optimization objectives"""
    resource: float      # Weight for resource cost
    latency: float       # Weight for latency penalty
    uncertainty: float   # Weight for uncertainty penalty
    stability: float     # Weight for stability penalty
    
    def __post_init__(self):
        """Validate weights sum to reasonable range"""
        total = self.resource + self.latency + self.uncertainty + self.stability
        if not (0.8 <= total <= 1.2):
            raise ValueError(f"Weights should sum to approximately 1.0, got {total}")
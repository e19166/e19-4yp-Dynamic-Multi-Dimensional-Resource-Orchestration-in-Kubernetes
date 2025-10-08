"""
Modular Resource Optimization Package

This package provides a modular approach to resource optimization for Kubernetes
with separated concerns and clean interfaces.

Main Components:
- OptimizationTypes: Core data types and enums
- StrategyManager: Optimization strategy management
- TraditionalOptimizer: Scipy-based optimization
- GaussianProcessOptimizer: Bayesian optimization with GP
- ResultAnalyzer: Result analysis and confidence assessment
- AdaptiveStrategy: Adaptive strategy selection
- MultiObjectiveOptimizer: Main optimization orchestrator
- AdaptiveOptimizer: High-level adaptive optimization

Usage:
    from resource_optimizer import create_adaptive_optimizer
    
    optimizer = create_adaptive_optimizer(latency_predictor)
    result = optimizer.optimize_resources(service_name, current_state, constraints)
"""

# Export main classes and factory functions
from .resource_optimizer import (
    MultiObjectiveOptimizer,
    AdaptiveOptimizer,
    create_multi_objective_optimizer,
    create_adaptive_optimizer
)

from .optimization_types import (
    OptimizationStrategy,
    ResourceConstraints,
    OptimizationResult,
    ServiceState,
    OptimizationWeights
)

from .strategy_manager import StrategyManager
from .traditional_optimizer import TraditionalOptimizer
from .gp_optimizer import GaussianProcessOptimizer
from .result_analyzer import ResultAnalyzer
from .adaptive_strategy import AdaptiveStrategy

__version__ = "2.0.0"
__author__ = "DARE Algorithm Team"

__all__ = [
    # Main classes
    'MultiObjectiveOptimizer',
    'AdaptiveOptimizer',
    
    # Factory functions
    'create_multi_objective_optimizer',
    'create_adaptive_optimizer',
    
    # Data types
    'OptimizationStrategy',
    'ResourceConstraints', 
    'OptimizationResult',
    'ServiceState',
    'OptimizationWeights',
    
    # Component classes
    'StrategyManager',
    'TraditionalOptimizer',
    'GaussianProcessOptimizer',
    'ResultAnalyzer',
    'AdaptiveStrategy'
]
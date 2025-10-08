"""
Advanced Resource Optimization Module for Kubernetes

This module implements intelligent resource allocation optimization that:
1. Minimizes resource usage while maintaining latency SLA
2. Uses multi-objective optimization considering cost, performance, and stability
3. Implements safety mechanisms to prevent resource starvation
4. Provides adaptive optimization strategies based on service behavior

This is a modular refactored version that separates concerns into focused components.
"""

import logging
from typing import Dict, Optional
import warnings
warnings.filterwarnings('ignore')

# Import modular components
from optimization_types import (
    OptimizationStrategy, ResourceConstraints, OptimizationResult, ServiceState
)
from strategy_manager import StrategyManager
from traditional_optimizer import TraditionalOptimizer
from gp_optimizer import GaussianProcessOptimizer
from result_analyzer import ResultAnalyzer
from adaptive_strategy import AdaptiveStrategy

logger = logging.getLogger(__name__)

class MultiObjectiveOptimizer:
    """
    Multi-objective optimization for resource allocation using modular components
    
    This class orchestrates the optimization process using specialized components:
    - StrategyManager: Handles optimization strategies and weights
    - TraditionalOptimizer: Scipy-based optimization methods
    - GaussianProcessOptimizer: Bayesian optimization with GP models
    - ResultAnalyzer: Result analysis and confidence assessment
    """
    
    def __init__(self, latency_predictor):
        self.latency_predictor = latency_predictor
        
        # Initialize modular components
        self.strategy_manager = StrategyManager()
        self.traditional_optimizer = TraditionalOptimizer(latency_predictor)
        self.gp_optimizer = GaussianProcessOptimizer(latency_predictor)
        self.result_analyzer = ResultAnalyzer(latency_predictor)
    
    def optimize_resources(self, service_name: str, current_state: Dict,
                          constraints: ResourceConstraints,
                          strategy: OptimizationStrategy = OptimizationStrategy.BALANCED,
                          use_gp: bool = True) -> OptimizationResult:
        """
        Optimize resource allocation for a service
        
        Args:
            service_name: Name of the service
            current_state: Current resource state (dict with keys: cpu_usage, memory_usage, etc.)
            constraints: Resource constraints
            strategy: Optimization strategy
            use_gp: Whether to use Gaussian Process for optimization
            
        Returns:
            Complete optimization result
        """
        
        try:
            # Convert current_state dict to ServiceState object
            service_state = self._create_service_state(service_name, current_state)
            
            # Get strategy weights
            weights = self.strategy_manager.get_strategy_weights(strategy)
            
            # Choose optimization method
            if use_gp and self.gp_optimizer.has_model(service_name):
                logger.info(f"Using GP optimization for {service_name}")
                result_dict = self.gp_optimizer.optimize(service_state, constraints, weights)
            else:
                logger.info(f"Using traditional optimization for {service_name}")
                result_dict = self.traditional_optimizer.optimize(service_state, constraints, weights)
            
            # Analyze and format result
            result = self.result_analyzer.analyze_result(
                result_dict, service_state, constraints, strategy.value
            )
            
            logger.info(f"Optimization completed for {service_name}: "
                       f"CPU: {result.cpu_limit:.3f}, Memory: {result.memory_limit/1e9:.2f}GB, "
                       f"Savings: {result.resource_savings:.1f}%, Risk: {result.risk_level}")
            
            return result
            
        except Exception as e:
            logger.error(f"Optimization failed for {service_name}: {e}")
            return self._create_fallback_result(current_state, strategy)
    
    def _create_service_state(self, service_name: str, current_state: Dict) -> ServiceState:
        """Convert dict to ServiceState object"""
        return ServiceState(
            service_name=service_name,
            cpu_usage=current_state['cpu_usage'],
            memory_usage=current_state['memory_usage'],
            cpu_limit=current_state['cpu_limit'],
            memory_limit=current_state['memory_limit'],
            request_rate=current_state['request_rate'],
            latency=current_state.get('latency')
        )
    
    def _create_fallback_result(self, current_state: Dict, strategy: OptimizationStrategy) -> OptimizationResult:
        """Create fallback result when optimization fails"""
        return OptimizationResult(
            cpu_limit=current_state['cpu_limit'],
            memory_limit=current_state['memory_limit'],
            expected_latency=current_state.get('latency', 0.5),
            latency_uncertainty=0.2,
            resource_savings=0.0,
            confidence=0.5,
            risk_level="HIGH",
            optimization_strategy=f"FALLBACK_{strategy.value}",
            iterations=0,
            convergence_status="FAILED"
        )
    
    def train_gp_model(self, service_name: str, training_data):
        """Train Gaussian Process model for a service"""
        return self.gp_optimizer.train_gp_model(service_name, training_data)
    
    def update_optimization_history(self, service_name: str, result: OptimizationResult,
                                   actual_latency: Optional[float] = None):
        """Update optimization history with results"""
        self.result_analyzer.update_optimization_history(service_name, result, actual_latency)
    
    def get_optimization_stats(self, service_name: str) -> Dict:
        """Get optimization statistics for a service"""
        return self.result_analyzer.get_service_performance_stats(service_name)
    
    def get_gp_model_info(self, service_name: str) -> Dict:
        """Get GP model information"""
        return self.gp_optimizer.get_model_info(service_name)


class AdaptiveOptimizer:
    """
    Adaptive optimization that learns from past results and adjusts strategy
    
    This is a high-level orchestrator that uses:
    - MultiObjectiveOptimizer: For actual optimization
    - AdaptiveStrategy: For strategy selection based on historical performance
    """
    
    def __init__(self, latency_predictor):
        self.base_optimizer = MultiObjectiveOptimizer(latency_predictor)
        self.adaptive_strategy = AdaptiveStrategy()
        
    def optimize_resources(self, service_name: str, current_state: Dict,
                          constraints: ResourceConstraints) -> OptimizationResult:
        """
        Adaptive resource optimization with automatic strategy selection
        
        Args:
            service_name: Name of the service
            current_state: Current resource state
            constraints: Resource constraints
            
        Returns:
            Optimization result
        """
        
        # Choose strategy based on historical performance
        strategy = self.adaptive_strategy.choose_strategy(service_name)
        
        # Perform optimization
        result = self.base_optimizer.optimize_resources(
            service_name, current_state, constraints, strategy
        )
        
        logger.info(f"Adaptive optimization for {service_name} used {strategy.value} strategy")
        
        return result
    
    def update_performance(self, service_name: str, result: OptimizationResult,
                          actual_latency: float):
        """Update performance metrics with actual results"""
        
        # Update both the base optimizer and adaptive strategy
        self.base_optimizer.update_optimization_history(service_name, result, actual_latency)
        self.adaptive_strategy.update_performance(service_name, result, actual_latency)
        
        logger.info(f"Updated performance metrics for {service_name}")
    
    def get_strategy_recommendations(self, service_name: str) -> Dict:
        """Get strategy recommendations for a service"""
        return self.adaptive_strategy.get_strategy_recommendations(service_name)
    
    def get_global_performance_summary(self) -> Dict:
        """Get global performance summary across all services"""
        return self.adaptive_strategy.get_global_performance_summary()
    
    def reset_service_history(self, service_name: str):
        """Reset performance history for a service"""
        self.adaptive_strategy.reset_service_history(service_name)
    
    def train_gp_model(self, service_name: str, training_data):
        """Train GP model for better optimization"""
        return self.base_optimizer.train_gp_model(service_name, training_data)


# Convenience factory functions for backward compatibility
def create_multi_objective_optimizer(latency_predictor):
    """Create a MultiObjectiveOptimizer instance"""
    return MultiObjectiveOptimizer(latency_predictor)


def create_adaptive_optimizer(latency_predictor):
    """Create an AdaptiveOptimizer instance"""
    return AdaptiveOptimizer(latency_predictor)


# Example usage and module information
if __name__ == "__main__":
    print("Modular Resource Optimization Module")
    print("=====================================")
    print("This refactored module provides:")
    print("- Modular architecture with separated concerns")
    print("- Strategy management (StrategyManager)")
    print("- Multiple optimization algorithms (Traditional & GP)")
    print("- Adaptive strategy selection based on performance")
    print("- Comprehensive result analysis and confidence assessment")
    print("- Easy extensibility and maintainability")
    print()
    print("Main classes:")
    print("- MultiObjectiveOptimizer: Core optimization with pluggable algorithms")
    print("- AdaptiveOptimizer: High-level adaptive optimization")
    print()
    print("Key improvements:")
    print("- Better separation of concerns")
    print("- Easier testing and debugging")
    print("- More maintainable codebase")
    print("- Extensible architecture for new optimization methods")
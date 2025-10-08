"""
Example usage of the modular resource optimization system

This example demonstrates how to use the refactored modular components
for resource optimization in Kubernetes services.
"""

import logging
from typing import Dict

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Example mock latency predictor for demonstration
class MockLatencyPredictor:
    """Mock latency predictor for example purposes"""
    
    def predict_latency(self, service_name: str, cpu_usage: float, 
                       memory_usage: float, cpu_limit: float, 
                       memory_limit: float, request_rate: float):
        """Mock latency prediction"""
        # Simple heuristic: latency increases with utilization
        cpu_util = cpu_usage / cpu_limit
        memory_util = memory_usage / memory_limit
        
        base_latency = 0.1  # 100ms base latency
        util_factor = max(cpu_util, memory_util)
        
        if util_factor > 0.9:
            latency = base_latency * (1 + (util_factor - 0.9) * 10)
        else:
            latency = base_latency * (1 + util_factor * 0.5)
        
        uncertainty = latency * 0.1  # 10% uncertainty
        confidence = 1.0 - uncertainty / latency
        
        return latency, uncertainty, confidence
    
    def get_service_risk_assessment(self, service_name: str, 
                                   cpu_util: float, memory_util: float):
        """Mock risk assessment"""
        max_util = max(cpu_util, memory_util)
        
        if max_util > 0.95:
            risk_level = "CRITICAL"
        elif max_util > 0.85:
            risk_level = "HIGH"
        elif max_util > 0.70:
            risk_level = "MEDIUM"
        else:
            risk_level = "LOW"
        
        return {"risk_level": risk_level}


def example_basic_optimization():
    """Example of basic optimization using MultiObjectiveOptimizer"""
    
    print("=== Basic Optimization Example ===")
    
    # Import the modular components
    from optimization_types import OptimizationStrategy, ResourceConstraints
    from resource_optimizer import create_multi_objective_optimizer
    
    # Create latency predictor and optimizer
    latency_predictor = MockLatencyPredictor()
    optimizer = create_multi_objective_optimizer(latency_predictor)
    
    # Define current service state
    current_state = {
        'cpu_usage': 0.8,      # 800m CPU currently used
        'memory_usage': 1.5e9, # 1.5GB memory currently used
        'cpu_limit': 1.0,      # 1 CPU limit
        'memory_limit': 2e9,   # 2GB memory limit
        'request_rate': 100    # 100 requests/sec
    }
    
    # Define constraints
    constraints = ResourceConstraints(
        min_cpu_ratio=1.1,
        max_cpu_ratio=1.5,
        min_memory_ratio=1.1,
        max_memory_ratio=1.5,
        latency_threshold=0.5,  # 500ms max latency
        safety_margin=0.15
    )
    
    # Optimize with different strategies
    strategies = [
        OptimizationStrategy.CONSERVATIVE,
        OptimizationStrategy.BALANCED,
        OptimizationStrategy.AGGRESSIVE
    ]
    
    for strategy in strategies:
        print(f"\n--- {strategy.value.upper()} Strategy ---")
        
        result = optimizer.optimize_resources(
            service_name="example-service",
            current_state=current_state,
            constraints=constraints,
            strategy=strategy,
            use_gp=False  # Use traditional optimization for this example
        )
        
        print(f"CPU Limit: {result.cpu_limit:.3f} cores")
        print(f"Memory Limit: {result.memory_limit/1e9:.2f} GB")
        print(f"Expected Latency: {result.expected_latency:.3f}s")
        print(f"Resource Savings: {result.resource_savings:.1f}%")
        print(f"Confidence: {result.confidence:.2f}")
        print(f"Risk Level: {result.risk_level}")
        print(f"Status: {result.convergence_status}")


def example_adaptive_optimization():
    """Example of adaptive optimization that learns from performance"""
    
    print("\n=== Adaptive Optimization Example ===")
    
    # Import adaptive optimizer
    from optimization_types import ResourceConstraints
    from resource_optimizer import create_adaptive_optimizer
    
    # Create adaptive optimizer
    latency_predictor = MockLatencyPredictor()
    adaptive_optimizer = create_adaptive_optimizer(latency_predictor)
    
    # Service state
    current_state = {
        'cpu_usage': 0.6,
        'memory_usage': 1.2e9,
        'cpu_limit': 1.0,
        'memory_limit': 2e9,
        'request_rate': 80
    }
    
    constraints = ResourceConstraints(latency_threshold=0.4)
    
    # Simulate multiple optimization cycles
    for cycle in range(3):
        print(f"\n--- Optimization Cycle {cycle + 1} ---")
        
        # Get strategy recommendation
        recommendations = adaptive_optimizer.get_strategy_recommendations("test-service")
        print(f"Strategy Recommendation: {recommendations}")
        
        # Perform optimization
        result = adaptive_optimizer.optimize_resources(
            service_name="test-service",
            current_state=current_state,
            constraints=constraints
        )
        
        print(f"Strategy Used: {result.optimization_strategy}")
        print(f"Resource Savings: {result.resource_savings:.1f}%")
        print(f"Confidence: {result.confidence:.2f}")
        
        # Simulate actual latency measurement and update performance
        simulated_actual_latency = result.expected_latency * (0.9 + 0.2 * (cycle / 3))
        adaptive_optimizer.update_performance("test-service", result, simulated_actual_latency)
        
        print(f"Actual Latency: {simulated_actual_latency:.3f}s")
        print(f"Prediction Accuracy: {abs(result.expected_latency - simulated_actual_latency):.3f}s difference")
    
    # Show final performance summary
    summary = adaptive_optimizer.get_global_performance_summary()
    print(f"\n--- Performance Summary ---")
    print(f"Services Managed: {summary['services_managed']}")
    print(f"Total Optimizations: {summary['total_optimizations']}")
    print(f"Success Rate: {summary['success_rate']:.1f}%")
    print(f"Total Savings: {summary['total_resource_savings']:.1f}%")


def example_component_usage():
    """Example of using individual components"""
    
    print("\n=== Individual Component Usage Example ===")
    
    # Import individual components
    from strategy_manager import StrategyManager
    from bounds_calculator import BoundsCalculator
    from optimization_types import OptimizationStrategy, ResourceConstraints, ServiceState
    
    # Example 1: Strategy Manager
    print("\n--- Strategy Manager ---")
    strategy_manager = StrategyManager()
    
    weights = strategy_manager.get_strategy_weights(OptimizationStrategy.AGGRESSIVE)
    print(f"Aggressive strategy weights: {weights}")
    
    # Example 2: Bounds Calculator
    print("\n--- Bounds Calculator ---")
    bounds_calc = BoundsCalculator()
    
    service_state = ServiceState(
        service_name="test",
        cpu_usage=0.7,
        memory_usage=1.5e9,
        cpu_limit=1.0,
        memory_limit=2e9,
        request_rate=100
    )
    
    constraints = ResourceConstraints()
    bounds = bounds_calc.calculate_bounds(service_state, constraints)
    print(f"Optimization bounds: CPU {bounds[0]}, Memory {bounds[1]}")
    
    is_valid = bounds_calc.validate_bounds(bounds, service_state)
    print(f"Bounds valid: {is_valid}")


if __name__ == "__main__":
    print("Modular Resource Optimization Examples")
    print("=====================================")
    
    # Run examples
    try:
        example_basic_optimization()
        example_adaptive_optimization()
        example_component_usage()
        
        print("\n=== All Examples Completed Successfully ===")
        
    except ImportError as e:
        print(f"Import error: {e}")
        print("Some dependencies may be missing (scipy, sklearn, numpy)")
        print("The modular structure works, but optimization requires these libraries")
        
    except Exception as e:
        print(f"Error running examples: {e}")
        print("This may be due to import path issues in the current environment")
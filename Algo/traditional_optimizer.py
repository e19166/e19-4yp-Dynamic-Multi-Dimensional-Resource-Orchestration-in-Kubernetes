"""
Traditional optimization algorithms using scipy
"""

import logging
from typing import Dict, List, Tuple, Optional
try:
    from scipy.optimize import minimize
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    
from optimization_types import (
    OptimizationStrategy, ResourceConstraints, OptimizationResult, 
    ServiceState, OptimizationWeights
)
from objective_calculator import ObjectiveCalculator
from bounds_calculator import BoundsCalculator

logger = logging.getLogger(__name__)


class TraditionalOptimizer:
    """
    Traditional optimization using scipy optimization methods
    """
    
    def __init__(self, latency_predictor):
        if not HAS_SCIPY:
            raise ImportError("scipy is required for traditional optimization")
            
        self.latency_predictor = latency_predictor
        self.objective_calculator = ObjectiveCalculator(latency_predictor)
        self.bounds_calculator = BoundsCalculator()
        self.optimization_methods = ['SLSQP', 'L-BFGS-B']
    
    def optimize(self, service_state: ServiceState,
                constraints: ResourceConstraints,
                weights: OptimizationWeights) -> Dict:
        """
        Perform traditional optimization
        
        Args:
            service_state: Current service state
            constraints: Resource constraints
            weights: Optimization weights
            
        Returns:
            Optimization result dictionary
        """
        
        # Calculate bounds
        bounds = self.bounds_calculator.calculate_bounds(service_state, constraints)
        
        if not self.bounds_calculator.validate_bounds(bounds, service_state):
            raise ValueError("Invalid optimization bounds")
        
        # Initial guess (current resources)
        x0 = [service_state.cpu_limit, service_state.memory_limit]
        
        # Try different optimization methods
        best_result = None
        best_cost = float('inf')
        
        for method in self.optimization_methods:
            try:
                result = self._optimize_with_method(
                    method, x0, bounds, service_state, constraints, weights
                )
                
                if result.success and result.fun < best_cost:
                    best_result = result
                    best_cost = result.fun
                    
            except Exception as e:
                logger.warning(f"Optimization method {method} failed: {e}")
                continue
        
        if best_result is None:
            raise ValueError("All optimization methods failed")
        
        return self._format_result(best_result, service_state)
    
    def _optimize_with_method(self, method: str, x0: List[float],
                            bounds: List[Tuple[float, float]],
                            service_state: ServiceState,
                            constraints: ResourceConstraints,
                            weights: OptimizationWeights):
        """Optimize with a specific scipy method"""
        
        if method == 'SLSQP':
            return self._optimize_slsqp(x0, bounds, service_state, constraints, weights)
        elif method == 'L-BFGS-B':
            return self._optimize_lbfgsb(x0, bounds, service_state, constraints, weights)
        else:
            raise ValueError(f"Unknown optimization method: {method}")
    
    def _optimize_slsqp(self, x0: List[float],
                       bounds: List[Tuple[float, float]],
                       service_state: ServiceState,
                       constraints: ResourceConstraints,
                       weights: OptimizationWeights):
        """Optimize using SLSQP (supports constraints)"""
        
        # Define objective function
        def objective_func(x):
            return self.objective_calculator.calculate_objective(
                tuple(x), service_state, constraints, weights
            )
        
        # Define constraint function
        def latency_constraint(x):
            return self.objective_calculator.calculate_latency_constraint(
                tuple(x), service_state, constraints.latency_threshold
            )
        
        # Optimization constraints
        cons = {'type': 'ineq', 'fun': latency_constraint}
        
        return minimize(
            fun=objective_func,
            x0=x0,
            method='SLSQP',
            bounds=bounds,
            constraints=cons,
            options={'maxiter': 100, 'ftol': 1e-6}
        )
    
    def _optimize_lbfgsb(self, x0: List[float],
                        bounds: List[Tuple[float, float]],
                        service_state: ServiceState,
                        constraints: ResourceConstraints,
                        weights: OptimizationWeights):
        """Optimize using L-BFGS-B (penalty method for constraints)"""
        
        def penalized_objective(x):
            # Base objective
            base_cost = self.objective_calculator.calculate_objective(
                tuple(x), service_state, constraints, weights
            )
            
            # Latency constraint penalty
            latency_violation = -self.objective_calculator.calculate_latency_constraint(
                tuple(x), service_state, constraints.latency_threshold
            )
            penalty = 1000 * max(0, latency_violation) ** 2
            
            return base_cost + penalty
        
        return minimize(
            fun=penalized_objective,
            x0=x0,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 100, 'ftol': 1e-6}
        )
    
    def _format_result(self, optimization_result, service_state: ServiceState) -> Dict:
        """Format optimization result"""
        
        cpu_opt, memory_opt = optimization_result.x
        
        # Get final latency prediction
        predicted_latency, uncertainty, _ = self.latency_predictor.predict_latency(
            service_name=service_state.service_name,
            cpu_usage=service_state.cpu_usage,
            memory_usage=service_state.memory_usage,
            cpu_limit=cpu_opt,
            memory_limit=memory_opt,
            request_rate=service_state.request_rate
        )
        
        return {
            'cpu_limit': cpu_opt,
            'memory_limit': memory_opt,
            'expected_latency': predicted_latency,
            'uncertainty': uncertainty,
            'iterations': optimization_result.nit,
            'status': 'CONVERGED' if optimization_result.success else 'FAILED',
            'objective_value': optimization_result.fun
        }
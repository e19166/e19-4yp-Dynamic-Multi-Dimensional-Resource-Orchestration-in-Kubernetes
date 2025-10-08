"""
Gaussian Process-based optimization (Bayesian optimization)
"""

import logging
from typing import Dict, List, Tuple
import pandas as pd
try:
    import numpy as np
    from scipy.optimize import differential_evolution
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF, ConstantKernel
    HAS_DEPENDENCIES = True
except ImportError:
    HAS_DEPENDENCIES = False
    
from optimization_types import (
    ResourceConstraints, ServiceState, OptimizationWeights
)
from bounds_calculator import BoundsCalculator

logger = logging.getLogger(__name__)


class GaussianProcessOptimizer:
    """
    Gaussian Process-based Bayesian optimization
    """
    
    def __init__(self, latency_predictor):
        if not HAS_DEPENDENCIES:
            raise ImportError("Required dependencies not available for GP optimization")
            
        self.latency_predictor = latency_predictor
        self.bounds_calculator = BoundsCalculator()
        self.gp_models = {}  # Store GP models for each service
    
    def optimize(self, service_state: ServiceState,
                constraints: ResourceConstraints,
                weights: OptimizationWeights) -> Dict:
        """
        Perform Gaussian Process-based optimization
        
        Args:
            service_state: Current service state
            constraints: Resource constraints
            weights: Optimization weights
            
        Returns:
            Optimization result dictionary
        """
        
        if service_state.service_name not in self.gp_models:
            raise ValueError(f"No GP model trained for service {service_state.service_name}")
        
        # Calculate bounds
        bounds = self.bounds_calculator.calculate_bounds(service_state, constraints)
        
        if not self.bounds_calculator.validate_bounds(bounds, service_state):
            raise ValueError("Invalid optimization bounds")
        
        # Get GP model
        gp_model = self.gp_models[service_state.service_name]
        
        # Define GP-based objective function
        def gp_objective(x):
            return self._calculate_gp_objective(x, gp_model, service_state, constraints, weights)
        
        # Use differential evolution for global optimization
        result = differential_evolution(
            func=gp_objective,
            bounds=bounds,
            maxiter=50,
            popsize=15,
            seed=42,
            atol=1e-6,
            tol=1e-6
        )
        
        if not result.success:
            raise ValueError("GP optimization failed to converge")
        
        return self._format_result(result, service_state)
    
    def _calculate_gp_objective(self, x: np.ndarray, gp_model,
                              service_state: ServiceState,
                              constraints: ResourceConstraints,
                              weights: OptimizationWeights) -> float:
        """Calculate objective using GP model predictions"""
        
        # Predict latency with GP model
        x_pred = np.array([x])
        latency_pred, latency_std = gp_model.predict(x_pred, return_std=True)
        
        predicted_latency = latency_pred[0]
        uncertainty = latency_std[0]
        
        # Calculate objective components
        resource_cost = self._calculate_resource_cost(x, service_state)
        latency_penalty = self._calculate_latency_penalty(predicted_latency, constraints.latency_threshold)
        uncertainty_penalty = uncertainty / constraints.latency_threshold
        
        # Combined objective (no stability penalty for GP as it handles exploration)
        total_cost = (weights.resource * resource_cost +
                     weights.latency * latency_penalty +
                     weights.uncertainty * uncertainty_penalty)
        
        return total_cost
    
    def _calculate_resource_cost(self, x: np.ndarray, service_state: ServiceState) -> float:
        """Calculate normalized resource cost"""
        cpu_limit, memory_limit = x
        cpu_cost = cpu_limit / service_state.cpu_limit
        memory_cost = memory_limit / service_state.memory_limit
        return (cpu_cost + memory_cost) / 2
    
    def _calculate_latency_penalty(self, predicted_latency: float, 
                                 latency_threshold: float) -> float:
        """Calculate latency penalty"""
        latency_violation = max(0, predicted_latency - latency_threshold)
        return latency_violation / latency_threshold
    
    def _format_result(self, optimization_result, service_state: ServiceState) -> Dict:
        """Format optimization result"""
        
        cpu_opt, memory_opt = optimization_result.x
        
        # Get final latency prediction using original predictor
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
    
    def train_gp_model(self, service_name: str, training_data: pd.DataFrame) -> bool:
        """
        Train Gaussian Process model for a service
        
        Args:
            service_name: Name of the service
            training_data: Historical data with columns ['cpu_limit', 'memory_limit', 'latency']
            
        Returns:
            True if training successful, False otherwise
        """
        
        if not HAS_DEPENDENCIES:
            logger.error("Cannot train GP model: required dependencies not available")
            return False
        
        if len(training_data) < 20:
            logger.warning(f"Insufficient data to train GP model for {service_name}")
            return False
        
        try:
            # Prepare features and targets
            required_columns = ['cpu_limit', 'memory_limit', 'latency']
            if not all(col in training_data.columns for col in required_columns):
                logger.error(f"Training data missing required columns: {required_columns}")
                return False
            
            features = training_data[['cpu_limit', 'memory_limit']].values
            targets = training_data['latency'].values
            
            # Create and train GP model
            kernel = ConstantKernel(1.0, (1e-3, 1e3)) * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2))
            gp_model = GaussianProcessRegressor(
                kernel=kernel,
                alpha=1e-6,
                normalize_y=True,
                n_restarts_optimizer=10,
                random_state=42
            )
            
            gp_model.fit(features, targets)
            
            # Store model
            self.gp_models[service_name] = gp_model
            
            logger.info(f"Trained GP model for {service_name} with {len(training_data)} samples")
            return True
            
        except Exception as e:
            logger.error(f"Failed to train GP model for {service_name}: {e}")
            return False
    
    def has_model(self, service_name: str) -> bool:
        """Check if GP model exists for service"""
        return service_name in self.gp_models
    
    def remove_model(self, service_name: str) -> bool:
        """Remove GP model for service"""
        if service_name in self.gp_models:
            del self.gp_models[service_name]
            logger.info(f"Removed GP model for {service_name}")
            return True
        return False
    
    def get_model_info(self, service_name: str) -> Dict:
        """Get information about GP model"""
        if service_name not in self.gp_models:
            return {}
        
        model = self.gp_models[service_name]
        return {
            'service_name': service_name,
            'kernel': str(model.kernel_),
            'log_marginal_likelihood': model.log_marginal_likelihood_value_,
            'n_training_samples': model.X_train_.shape[0] if hasattr(model, 'X_train_') else 0
        }
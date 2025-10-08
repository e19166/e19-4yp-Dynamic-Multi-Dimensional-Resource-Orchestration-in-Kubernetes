"""
Strategy management for optimization algorithms
"""

import logging
from typing import Dict
from optimization_types import OptimizationStrategy, OptimizationWeights

logger = logging.getLogger(__name__)


class StrategyManager:
    """
    Manages optimization strategies and their configurations
    """
    
    def __init__(self):
        self._strategy_configs = self._initialize_strategies()
    
    def _initialize_strategies(self) -> Dict[OptimizationStrategy, OptimizationWeights]:
        """Initialize default strategy configurations"""
        return {
            OptimizationStrategy.CONSERVATIVE: OptimizationWeights(
                resource=0.3,      # Lower weight on resource saving
                latency=0.4,       # High weight on latency
                uncertainty=0.2,   # Consider uncertainty
                stability=0.1      # Prefer stability
            ),
            OptimizationStrategy.AGGRESSIVE: OptimizationWeights(
                resource=0.6,      # High weight on resource saving
                latency=0.3,       # Moderate latency concern
                uncertainty=0.05,  # Lower uncertainty concern
                stability=0.05     # Allow more changes
            ),
            OptimizationStrategy.BALANCED: OptimizationWeights(
                resource=0.4,
                latency=0.35,
                uncertainty=0.15,
                stability=0.1
            ),
            OptimizationStrategy.ADAPTIVE: OptimizationWeights(
                resource=0.4,      # Default to balanced
                latency=0.35,
                uncertainty=0.15,
                stability=0.1
            )
        }
    
    def get_strategy_weights(self, strategy: OptimizationStrategy) -> OptimizationWeights:
        """Get optimization weights for a strategy"""
        return self._strategy_configs.get(strategy, self._strategy_configs[OptimizationStrategy.BALANCED])
    
    def update_strategy_weights(self, strategy: OptimizationStrategy, weights: OptimizationWeights):
        """Update weights for a strategy"""
        self._strategy_configs[strategy] = weights
        logger.info(f"Updated weights for strategy {strategy.value}")
    
    def get_all_strategies(self) -> Dict[OptimizationStrategy, OptimizationWeights]:
        """Get all strategy configurations"""
        return self._strategy_configs.copy()
    
    def validate_strategy(self, strategy: OptimizationStrategy) -> bool:
        """Validate if strategy is supported"""
        return strategy in self._strategy_configs
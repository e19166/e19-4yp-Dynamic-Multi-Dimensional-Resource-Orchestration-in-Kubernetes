"""
Adaptive optimization strategy management
"""

import logging
from typing import Dict
import pandas as pd
from optimization_types import OptimizationStrategy, OptimizationResult
from strategy_manager import StrategyManager

logger = logging.getLogger(__name__)


class AdaptiveStrategy:
    """
    Manages adaptive strategy selection based on historical performance
    """
    
    def __init__(self):
        self.strategy_manager = StrategyManager()
        self.service_strategies = {}  # Current strategy for each service
        self.performance_metrics = {}  # Performance history per service
    
    def choose_strategy(self, service_name: str) -> OptimizationStrategy:
        """
        Choose optimization strategy based on historical performance
        
        Args:
            service_name: Name of the service
            
        Returns:
            Best strategy for the service
        """
        
        if service_name not in self.performance_metrics:
            logger.info(f"No history for {service_name}, using BALANCED strategy")
            return OptimizationStrategy.BALANCED
        
        metrics = self.performance_metrics[service_name]
        if len(metrics) < 3:  # Need some history
            return OptimizationStrategy.BALANCED
        
        # Calculate success rates for each strategy
        strategy_performance = self._calculate_strategy_performance(metrics)
        
        # Choose best strategy
        best_strategy = self._select_best_strategy(strategy_performance)
        
        # Update current strategy
        self.service_strategies[service_name] = best_strategy
        
        logger.info(f"Chose {best_strategy.value} strategy for {service_name}")
        return best_strategy
    
    def _calculate_strategy_performance(self, metrics: list) -> Dict[OptimizationStrategy, float]:
        """Calculate performance metrics for each strategy"""
        
        strategy_performance = {}
        
        for strategy in OptimizationStrategy:
            if strategy == OptimizationStrategy.ADAPTIVE:
                continue  # Skip adaptive as it's the meta-strategy
            
            # Get metrics for this strategy
            strategy_metrics = [m for m in metrics if m['strategy'] == strategy.value]
            
            if not strategy_metrics:
                strategy_performance[strategy] = 0.5  # Neutral prior
                continue
            
            # Calculate composite score
            success_rate = sum(1 for m in strategy_metrics if m.get('success', False)) / len(strategy_metrics)
            avg_savings = sum(m.get('resource_savings', 0) for m in strategy_metrics) / len(strategy_metrics)
            
            # Weight recent performance more heavily
            recent_metrics = strategy_metrics[-5:]  # Last 5 attempts
            recent_success_rate = (sum(1 for m in recent_metrics if m.get('success', False)) / 
                                 len(recent_metrics) if recent_metrics else 0)
            
            # Composite score: 60% success rate, 20% savings, 20% recent performance
            composite_score = (0.6 * success_rate + 
                             0.2 * min(1.0, avg_savings / 20.0) +  # Normalize savings
                             0.2 * recent_success_rate)
            
            strategy_performance[strategy] = composite_score
        
        return strategy_performance
    
    def _select_best_strategy(self, strategy_performance: Dict[OptimizationStrategy, float]) -> OptimizationStrategy:
        """Select the best performing strategy"""
        
        if not strategy_performance:
            return OptimizationStrategy.BALANCED
        
        # Find strategy with highest performance
        best_strategy = max(strategy_performance.keys(), 
                           key=lambda s: strategy_performance[s])
        
        # Add some exploration: occasionally try other strategies
        best_score = strategy_performance[best_strategy]
        
        # If best score is not significantly better, stick with balanced
        if best_score < 0.7:
            return OptimizationStrategy.BALANCED
        
        return best_strategy
    
    def update_performance(self, service_name: str, result: OptimizationResult,
                          actual_latency: float):
        """
        Update performance metrics with actual results
        
        Args:
            service_name: Name of the service
            result: Optimization result
            actual_latency: Actual measured latency
        """
        
        if service_name not in self.performance_metrics:
            self.performance_metrics[service_name] = []
        
        # Determine success based on latency prediction accuracy
        success = actual_latency <= result.expected_latency * 1.2  # 20% tolerance
        
        performance_entry = {
            'timestamp': pd.Timestamp.now(),
            'strategy': result.optimization_strategy,
            'predicted_latency': result.expected_latency,
            'actual_latency': actual_latency,
            'success': success,
            'resource_savings': result.resource_savings,
            'confidence': result.confidence,
            'risk_level': result.risk_level
        }
        
        self.performance_metrics[service_name].append(performance_entry)
        
        # Keep only recent performance data (last 50 entries)
        if len(self.performance_metrics[service_name]) > 50:
            self.performance_metrics[service_name] = self.performance_metrics[service_name][-50:]
        
        logger.info(f"Updated performance for {service_name}: "
                   f"Success={success}, Strategy={result.optimization_strategy}, "
                   f"Savings={result.resource_savings:.1f}%")
    
    def get_strategy_recommendations(self, service_name: str) -> Dict:
        """Get strategy recommendations and analysis for a service"""
        
        if service_name not in self.performance_metrics:
            return {
                'current_strategy': 'BALANCED',
                'recommendation': 'No historical data available',
                'confidence': 0.5
            }
        
        metrics = self.performance_metrics[service_name]
        strategy_performance = self._calculate_strategy_performance(metrics)
        
        # Current strategy
        current_strategy = self.service_strategies.get(service_name, OptimizationStrategy.BALANCED)
        
        # Best strategy
        best_strategy = self._select_best_strategy(strategy_performance)
        
        # Generate recommendation
        recommendation = self._generate_recommendation(current_strategy, best_strategy, strategy_performance)
        
        return {
            'current_strategy': current_strategy.value,
            'recommended_strategy': best_strategy.value,
            'recommendation': recommendation,
            'strategy_scores': {s.value: score for s, score in strategy_performance.items()},
            'total_optimizations': len(metrics),
            'confidence': strategy_performance.get(best_strategy, 0.5)
        }
    
    def _generate_recommendation(self, current: OptimizationStrategy, 
                               best: OptimizationStrategy,
                               performance: Dict[OptimizationStrategy, float]) -> str:
        """Generate human-readable recommendation"""
        
        if current == best:
            score = performance.get(current, 0.5)
            if score > 0.8:
                return f"Continue with {current.value} strategy (performing well)"
            else:
                return f"Continue with {current.value} strategy (moderate performance)"
        else:
            current_score = performance.get(current, 0.5)
            best_score = performance.get(best, 0.5)
            improvement = (best_score - current_score) * 100
            
            return (f"Switch to {best.value} strategy "
                   f"(potential {improvement:.1f}% improvement)")
    
    def reset_service_history(self, service_name: str):
        """Reset performance history for a service"""
        if service_name in self.performance_metrics:
            del self.performance_metrics[service_name]
        if service_name in self.service_strategies:
            del self.service_strategies[service_name]
        logger.info(f"Reset history for {service_name}")
    
    def get_global_performance_summary(self) -> Dict:
        """Get summary of global performance across all services"""
        
        total_optimizations = 0
        total_successes = 0
        total_savings = 0.0
        services_managed = len(self.performance_metrics)
        
        strategy_counts = {s.value: 0 for s in OptimizationStrategy}
        
        for service_name, metrics in self.performance_metrics.items():
            total_optimizations += len(metrics)
            total_successes += sum(1 for m in metrics if m.get('success', False))
            total_savings += sum(m.get('resource_savings', 0) for m in metrics)
            
            # Count strategy usage
            for metric in metrics:
                strategy = metric.get('strategy', 'UNKNOWN')
                if strategy in strategy_counts:
                    strategy_counts[strategy] += 1
        
        success_rate = (total_successes / total_optimizations * 100) if total_optimizations > 0 else 0
        
        return {
            'services_managed': services_managed,
            'total_optimizations': total_optimizations,
            'success_rate': success_rate,
            'total_resource_savings': total_savings,
            'average_savings_per_optimization': total_savings / max(total_optimizations, 1),
            'strategy_usage': strategy_counts
        }
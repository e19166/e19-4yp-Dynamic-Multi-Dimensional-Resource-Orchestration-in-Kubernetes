"""
Advanced Latency Prediction Module for Kubernetes Services

This module implements sophisticated latency prediction using:
1. Service-specific behavioral models
2. Critical Reduction Point (CRP) analysis
3. Workload pattern recognition
4. Multi-variate latency modeling
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import logging
from collections import deque
from enum import Enum

logger = logging.getLogger(__name__)

class WorkloadPattern(Enum):
    """Different workload patterns observed in the system"""
    STEADY = "steady"           # Consistent load
    BURSTY = "bursty"          # Intermittent high load
    GRADUAL_INCREASE = "gradual_increase"  # Slowly increasing load
    VOLATILE = "volatile"       # Highly variable load

@dataclass
class LatencyProfile:
    """Service latency characteristics"""
    service_name: str
    service_type: str
    cpu_sensitivity: float      # How much CPU affects latency
    memory_sensitivity: float   # How much memory affects latency
    crp_cpu: float             # Critical Reduction Point for CPU
    crp_memory: float          # Critical Reduction Point for memory
    base_latency: float        # Baseline latency under normal conditions
    latency_threshold: float   # Maximum acceptable latency
    volatility_factor: float   # How volatile the service latency is

class LatencyPredictor:
    """
    Advanced latency prediction using machine learning and behavioral modeling
    """
    
    def __init__(self):
        self.service_profiles = {}
        self.ml_models = {}
        self.scalers = {}
        self.feature_history = {}
        self.prediction_history = deque(maxlen=1000)
        
        # Initialize default service profiles based on research findings
        self._initialize_default_profiles()
        
    def _initialize_default_profiles(self):
        """Initialize service profiles based on experimental analysis"""
        
        # CPU-bound services (Prime Verifier, Hash Generator)
        self.service_profiles['cpu_bound'] = LatencyProfile(
            service_name="generic_cpu_bound",
            service_type="cpu_bound",
            cpu_sensitivity=0.85,      # High CPU sensitivity
            memory_sensitivity=0.15,   # Low memory sensitivity
            crp_cpu=0.70,             # CRP at 70% CPU utilization
            crp_memory=0.85,          # Higher memory tolerance
            base_latency=0.1,         # 100ms baseline
            latency_threshold=0.5,    # 500ms max acceptable
            volatility_factor=0.3     # Moderate volatility
        )
        
        # Memory-resilient services (Echo, Password Generator)
        self.service_profiles['memory_resilient'] = LatencyProfile(
            service_name="generic_memory_resilient",
            service_type="memory_resilient",
            cpu_sensitivity=0.30,
            memory_sensitivity=0.10,   # Very low memory sensitivity
            crp_cpu=0.90,             # High CPU tolerance
            crp_memory=0.95,          # Very high memory tolerance
            base_latency=0.05,        # 50ms baseline
            latency_threshold=0.3,    # 300ms max acceptable
            volatility_factor=0.1     # Low volatility
        )
        
        # Balanced services
        self.service_profiles['balanced'] = LatencyProfile(
            service_name="generic_balanced",
            service_type="balanced",
            cpu_sensitivity=0.60,
            memory_sensitivity=0.40,
            crp_cpu=0.75,
            crp_memory=0.80,
            base_latency=0.08,        # 80ms baseline
            latency_threshold=0.4,    # 400ms max acceptable
            volatility_factor=0.2     # Moderate volatility
        )
    
    def register_service(self, service_name: str, service_type: str, 
                        custom_profile: Optional[LatencyProfile] = None):
        """Register a service with its latency profile"""
        
        if custom_profile:
            self.service_profiles[service_name] = custom_profile
        else:
            # Use default profile based on service type
            default_profile = self.service_profiles.get(service_type)
            if default_profile:
                # Create a copy with the specific service name
                self.service_profiles[service_name] = LatencyProfile(
                    service_name=service_name,
                    service_type=default_profile.service_type,
                    cpu_sensitivity=default_profile.cpu_sensitivity,
                    memory_sensitivity=default_profile.memory_sensitivity,
                    crp_cpu=default_profile.crp_cpu,
                    crp_memory=default_profile.crp_memory,
                    base_latency=default_profile.base_latency,
                    latency_threshold=default_profile.latency_threshold,
                    volatility_factor=default_profile.volatility_factor
                )
        
        # Initialize feature history for this service
        self.feature_history[service_name] = deque(maxlen=100)
        
        logger.info(f"Registered service {service_name} with type {service_type}")
    
    def detect_workload_pattern(self, service_name: str, 
                               recent_metrics: List[Dict]) -> WorkloadPattern:
        """Detect current workload pattern for better prediction"""
        
        if len(recent_metrics) < 10:
            return WorkloadPattern.STEADY
        
        # Extract request rates and latencies
        request_rates = [m.get('request_rate', 0) for m in recent_metrics]
        latencies = [m.get('latency', 0) for m in recent_metrics]
        
        # Calculate statistics
        rate_cv = np.std(request_rates) / (np.mean(request_rates) + 1e-6)  # Coefficient of variation
        latency_cv = np.std(latencies) / (np.mean(latencies) + 1e-6)
        
        # Detect trends
        rate_trend = np.polyfit(range(len(request_rates)), request_rates, 1)[0]
        
        # Classify pattern
        if rate_cv > 0.5 or latency_cv > 0.4:
            return WorkloadPattern.VOLATILE
        elif rate_trend > np.mean(request_rates) * 0.1:  # 10% increase trend
            return WorkloadPattern.GRADUAL_INCREASE
        elif max(request_rates) > np.mean(request_rates) * 2:  # Bursts detected
            return WorkloadPattern.BURSTY
        else:
            return WorkloadPattern.STEADY
    
    def predict_latency_analytical(self, service_name: str, 
                                 cpu_usage: float, memory_usage: float,
                                 cpu_limit: float, memory_limit: float,
                                 request_rate: float) -> Tuple[float, float]:
        """
        Analytical latency prediction based on service profile and CRP analysis
        
        Returns:
            Tuple of (predicted_latency, confidence)
        """
        
        if service_name not in self.service_profiles:
            logger.warning(f"No profile found for {service_name}, using balanced profile")
            profile = self.service_profiles['balanced']
        else:
            profile = self.service_profiles[service_name]
        
        # Calculate resource utilization
        cpu_utilization = cpu_usage / max(cpu_limit, 0.001)
        memory_utilization = memory_usage / max(memory_limit, 0.001)
        
        # Base latency adjusted for current load
        base_latency = profile.base_latency * (1 + 0.1 * np.log(1 + request_rate))
        
        # Calculate resource pressure beyond CRP
        cpu_pressure = max(0, cpu_utilization - profile.crp_cpu)
        memory_pressure = max(0, memory_utilization - profile.crp_memory)
        
        # Exponential latency increase beyond CRP (key finding from research)
        if cpu_pressure > 0:
            cpu_impact = profile.cpu_sensitivity * (np.exp(cpu_pressure * 8) - 1)
        else:
            # Linear increase before CRP
            cpu_impact = profile.cpu_sensitivity * cpu_utilization * 0.2
        
        if memory_pressure > 0:
            memory_impact = profile.memory_sensitivity * (np.exp(memory_pressure * 5) - 1)
        else:
            memory_impact = profile.memory_sensitivity * memory_utilization * 0.1
        
        # Combined latency prediction
        predicted_latency = base_latency * (1 + cpu_impact + memory_impact)
        
        # Add volatility based on service characteristics
        volatility_factor = profile.volatility_factor * np.random.normal(0, 0.1)
        predicted_latency *= (1 + volatility_factor)
        
        # Calculate confidence based on how close we are to CRP
        cpu_safety = max(0, 1 - (cpu_utilization / profile.crp_cpu))
        memory_safety = max(0, 1 - (memory_utilization / profile.crp_memory))
        confidence = min(cpu_safety, memory_safety)
        
        return max(0, predicted_latency), confidence
    
    def train_ml_model(self, service_name: str, training_data: pd.DataFrame):
        """
        Train ML model for more accurate latency prediction
        
        Expected columns: cpu_usage, memory_usage, cpu_limit, memory_limit, 
                         request_rate, latency, timestamp
        """
        
        if len(training_data) < 50:
            logger.warning(f"Insufficient training data for {service_name}: {len(training_data)} samples")
            return
        
        # Feature engineering
        features = self._engineer_features(training_data)
        target = training_data['latency'].values
        
        # Split data temporally
        split_idx = int(len(features) * 0.8)
        X_train, X_test = features[:split_idx], features[split_idx:]
        y_train, y_test = target[:split_idx], target[split_idx:]
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train Random Forest model
        model = RandomForestRegressor(
            n_estimators=200,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
        
        model.fit(X_train_scaled, y_train)
        
        # Evaluate
        y_pred = model.predict(X_test_scaled)
        r2 = r2_score(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        
        logger.info(f"Trained ML model for {service_name}: R² = {r2:.4f}, MSE = {mse:.6f}")
        
        # Store model and scaler
        self.ml_models[service_name] = model
        self.scalers[service_name] = scaler
        
        # Save to disk
        joblib.dump(model, f"models/{service_name}_latency_model.pkl")
        joblib.dump(scaler, f"models/{service_name}_latency_scaler.pkl")
    
    def _engineer_features(self, data: pd.DataFrame) -> np.ndarray:
        """Engineer features for ML model"""
        
        features = []
        
        # Basic resource features
        features.extend([
            data['cpu_usage'].values,
            data['memory_usage'].values,
            data['request_rate'].values,
            data['cpu_limit'].values,
            data['memory_limit'].values
        ])
        
        # Utilization ratios
        features.extend([
            data['cpu_usage'] / data['cpu_limit'],
            data['memory_usage'] / data['memory_limit']
        ])
        
        # Temporal features
        if 'timestamp' in data.columns:
            data['hour'] = pd.to_datetime(data['timestamp']).dt.hour
            features.extend([
                np.sin(2 * np.pi * data['hour'] / 24),  # Hour cyclical
                np.cos(2 * np.pi * data['hour'] / 24)
            ])
        
        # Rolling statistics (if enough data)
        if len(data) > 5:
            features.extend([
                data['request_rate'].rolling(5, min_periods=1).mean().values,
                data['request_rate'].rolling(5, min_periods=1).std().fillna(0).values,
                data['latency'].shift(1).fillna(data['latency'].mean()).values  # Previous latency
            ])
        else:
            features.extend([
                data['request_rate'].values,
                np.zeros(len(data)),
                np.full(len(data), data['latency'].mean())
            ])
        
        return np.column_stack(features)
    
    def predict_latency_ml(self, service_name: str, 
                          cpu_usage: float, memory_usage: float,
                          cpu_limit: float, memory_limit: float,
                          request_rate: float, 
                          previous_latency: float = None) -> Tuple[float, float]:
        """
        ML-based latency prediction with uncertainty estimation
        
        Returns:
            Tuple of (predicted_latency, uncertainty)
        """
        
        if service_name not in self.ml_models:
            logger.warning(f"No ML model found for {service_name}, using analytical prediction")
            return self.predict_latency_analytical(
                service_name, cpu_usage, memory_usage, cpu_limit, memory_limit, request_rate
            )
        
        model = self.ml_models[service_name]
        scaler = self.scalers[service_name]
        
        # Prepare features
        current_time = pd.Timestamp.now()
        hour = current_time.hour
        
        # Get previous latency from history or use default
        if previous_latency is None and self.feature_history[service_name]:
            previous_latency = self.feature_history[service_name][-1].get('latency', 0.1)
        elif previous_latency is None:
            previous_latency = 0.1
        
        features = np.array([[
            cpu_usage,
            memory_usage,
            request_rate,
            cpu_limit,
            memory_limit,
            cpu_usage / max(cpu_limit, 0.001),
            memory_usage / max(memory_limit, 0.001),
            np.sin(2 * np.pi * hour / 24),
            np.cos(2 * np.pi * hour / 24),
            request_rate,  # Rolling mean (same as current for single prediction)
            0,  # Rolling std (0 for single prediction)
            previous_latency
        ]])
        
        # Scale and predict
        features_scaled = scaler.transform(features)
        
        # For uncertainty estimation, use ensemble predictions
        if hasattr(model, 'estimators_'):
            # Random Forest - use tree predictions for uncertainty
            tree_predictions = [tree.predict(features_scaled)[0] for tree in model.estimators_]
            predicted_latency = np.mean(tree_predictions)
            uncertainty = np.std(tree_predictions)
        else:
            predicted_latency = model.predict(features_scaled)[0]
            uncertainty = 0.1  # Default uncertainty
        
        return max(0, predicted_latency), uncertainty
    
    def predict_latency(self, service_name: str,
                       cpu_usage: float, memory_usage: float,
                       cpu_limit: float, memory_limit: float,
                       request_rate: float,
                       use_ml: bool = True) -> Tuple[float, float, str]:
        """
        Unified latency prediction interface
        
        Returns:
            Tuple of (predicted_latency, confidence/uncertainty, method_used)
        """
        
        # Store current metrics in history
        if service_name in self.feature_history:
            self.feature_history[service_name].append({
                'cpu_usage': cpu_usage,
                'memory_usage': memory_usage,
                'cpu_limit': cpu_limit,
                'memory_limit': memory_limit,
                'request_rate': request_rate,
                'timestamp': pd.Timestamp.now()
            })
        
        if use_ml and service_name in self.ml_models:
            predicted_latency, uncertainty = self.predict_latency_ml(
                service_name, cpu_usage, memory_usage, cpu_limit, memory_limit, request_rate
            )
            method = "ML"
        else:
            predicted_latency, confidence = self.predict_latency_analytical(
                service_name, cpu_usage, memory_usage, cpu_limit, memory_limit, request_rate
            )
            uncertainty = 1 - confidence  # Convert confidence to uncertainty
            method = "Analytical"
        
        # Store prediction in history
        self.prediction_history.append({
            'service_name': service_name,
            'predicted_latency': predicted_latency,
            'uncertainty': uncertainty,
            'method': method,
            'timestamp': pd.Timestamp.now()
        })
        
        return predicted_latency, uncertainty, method
    
    def update_profile_from_data(self, service_name: str, 
                                historical_data: pd.DataFrame):
        """
        Update service profile based on observed data
        (Adaptive learning of service characteristics)
        """
        
        if service_name not in self.service_profiles:
            logger.warning(f"Service {service_name} not registered")
            return
        
        if len(historical_data) < 20:
            logger.warning(f"Insufficient data to update profile for {service_name}")
            return
        
        profile = self.service_profiles[service_name]
        
        # Analyze CPU sensitivity
        cpu_correlations = []
        for i in range(1, len(historical_data)):
            cpu_change = (historical_data.iloc[i]['cpu_usage'] / historical_data.iloc[i]['cpu_limit']) - \
                        (historical_data.iloc[i-1]['cpu_usage'] / historical_data.iloc[i-1]['cpu_limit'])
            latency_change = historical_data.iloc[i]['latency'] - historical_data.iloc[i-1]['latency']
            
            if cpu_change != 0:
                cpu_correlations.append(latency_change / cpu_change)
        
        if cpu_correlations:
            new_cpu_sensitivity = np.median(cpu_correlations)
            # Smooth update
            profile.cpu_sensitivity = 0.8 * profile.cpu_sensitivity + 0.2 * abs(new_cpu_sensitivity)
        
        # Update CRP based on observed latency spikes
        high_latency_data = historical_data[historical_data['latency'] > profile.latency_threshold * 0.8]
        if len(high_latency_data) > 0:
            crp_cpu_candidates = high_latency_data['cpu_usage'] / high_latency_data['cpu_limit']
            if len(crp_cpu_candidates) > 0:
                new_crp_cpu = np.percentile(crp_cpu_candidates, 20)  # 20th percentile
                profile.crp_cpu = 0.9 * profile.crp_cpu + 0.1 * new_crp_cpu
        
        logger.info(f"Updated profile for {service_name}: "
                   f"CPU sensitivity = {profile.cpu_sensitivity:.3f}, "
                   f"CRP CPU = {profile.crp_cpu:.3f}")
    
    def get_service_risk_assessment(self, service_name: str,
                                  current_cpu_util: float,
                                  current_memory_util: float) -> Dict:
        """Get risk assessment for current resource utilization"""
        
        if service_name not in self.service_profiles:
            return {"error": "Service not registered"}
        
        profile = self.service_profiles[service_name]
        
        # Calculate distances to CRP
        cpu_distance_to_crp = profile.crp_cpu - current_cpu_util
        memory_distance_to_crp = profile.crp_memory - current_memory_util
        
        # Risk levels
        if cpu_distance_to_crp < 0.05 or memory_distance_to_crp < 0.05:
            risk_level = "CRITICAL"
        elif cpu_distance_to_crp < 0.15 or memory_distance_to_crp < 0.15:
            risk_level = "HIGH"
        elif cpu_distance_to_crp < 0.25 or memory_distance_to_crp < 0.25:
            risk_level = "MEDIUM"
        else:
            risk_level = "LOW"
        
        return {
            "risk_level": risk_level,
            "cpu_distance_to_crp": cpu_distance_to_crp,
            "memory_distance_to_crp": memory_distance_to_crp,
            "cpu_utilization": current_cpu_util,
            "memory_utilization": current_memory_util,
            "service_type": profile.service_type
        }

# Example usage
if __name__ == "__main__":
    # Initialize predictor
    predictor = LatencyPredictor()
    
    # Register services
    predictor.register_service("hash-gen-service", "cpu_bound")
    predictor.register_service("echo-service", "memory_resilient")
    
    # Predict latency
    latency, uncertainty, method = predictor.predict_latency(
        service_name="hash-gen-service",
        cpu_usage=0.8,
        memory_usage=500000000,
        cpu_limit=1.0,
        memory_limit=1000000000,
        request_rate=50
    )
    
    print(f"Predicted latency: {latency:.4f}s (±{uncertainty:.4f}) using {method}")
    
    # Risk assessment
    risk = predictor.get_service_risk_assessment(
        "hash-gen-service", 
        current_cpu_util=0.75, 
        current_memory_util=0.5
    )
    print(f"Risk assessment: {risk}")
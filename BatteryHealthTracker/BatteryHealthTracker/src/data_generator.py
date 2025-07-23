import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

class EVBatteryDataGenerator:
    """
    Generates realistic electric vehicle battery telemetry data for different vehicle types
    """
    
    def __init__(self):
        self.vehicle_types = {
            'car': {
                'battery_capacity': (40, 100),  # kWh
                'degradation_rate': (0.02, 0.05),  # % per year
                'usage_pattern': 'urban',
                'temperature_sensitivity': 0.8
            },
            'bus': {
                'battery_capacity': (200, 400),  # kWh
                'degradation_rate': (0.03, 0.08),  # % per year
                'usage_pattern': 'heavy_duty',
                'temperature_sensitivity': 1.2
            },
            'truck': {
                'battery_capacity': (300, 600),  # kWh
                'degradation_rate': (0.04, 0.09),  # % per year
                'usage_pattern': 'commercial',
                'temperature_sensitivity': 1.1
            },
            'motorcycle': {
                'battery_capacity': (5, 20),  # kWh
                'degradation_rate': (0.01, 0.04),  # % per year
                'usage_pattern': 'light_duty',
                'temperature_sensitivity': 0.6
            }
        }
    
    def generate_vehicle_profile(self, vehicle_type, vehicle_id):
        """Generate a specific vehicle profile"""
        config = self.vehicle_types[vehicle_type]
        
        return {
            'vehicle_id': vehicle_id,
            'vehicle_type': vehicle_type,
            'battery_capacity': np.random.uniform(*config['battery_capacity']),
            'degradation_rate': np.random.uniform(*config['degradation_rate']),
            'usage_pattern': config['usage_pattern'],
            'temperature_sensitivity': config['temperature_sensitivity'],
            'manufacturing_date': datetime.now() - timedelta(days=np.random.randint(30, 1460)),  # 1 month to 4 years old
            'initial_resistance': np.random.uniform(0.1, 0.3),  # Ohms
            'cycle_life_expected': np.random.randint(2000, 8000)
        }
    
    def generate_battery_telemetry(self, vehicle_profile, num_points=1000):
        """Generate battery telemetry data for a specific vehicle"""
        data = []
        
        start_date = vehicle_profile['manufacturing_date']
        end_date = datetime.now()
        
        # Calculate age in days
        age_days = (end_date - start_date).days
        
        for i in range(num_points):
            # Time progression
            current_time = start_date + timedelta(days=(age_days * i / num_points))
            age_years = (current_time - start_date).days / 365.25
            
            # Base degradation based on age and usage
            base_degradation = vehicle_profile['degradation_rate'] * age_years
            
            # Environmental factors
            temperature = np.random.normal(25, 10)  # Celsius
            humidity = np.random.uniform(30, 90)  # %
            
            # Temperature impact on degradation
            temp_factor = 1 + (temperature - 25) * 0.01 * vehicle_profile['temperature_sensitivity']
            
            # Usage patterns
            if vehicle_profile['usage_pattern'] == 'heavy_duty':
                daily_cycles = np.random.uniform(2, 5)
                depth_of_discharge = np.random.uniform(0.7, 0.95)
            elif vehicle_profile['usage_pattern'] == 'commercial':
                daily_cycles = np.random.uniform(1, 3)
                depth_of_discharge = np.random.uniform(0.6, 0.9)
            elif vehicle_profile['usage_pattern'] == 'urban':
                daily_cycles = np.random.uniform(0.5, 2)
                depth_of_discharge = np.random.uniform(0.4, 0.8)
            else:  # light_duty
                daily_cycles = np.random.uniform(0.3, 1.5)
                depth_of_discharge = np.random.uniform(0.3, 0.7)
            
            # Calculate cycle count
            total_cycles = daily_cycles * (current_time - start_date).days
            
            # Capacity degradation
            cycle_degradation = (total_cycles / vehicle_profile['cycle_life_expected']) * 20  # % degradation
            
            # State of Health calculation
            soh = max(70, 100 - base_degradation - cycle_degradation * temp_factor + np.random.normal(0, 2))
            
            # Internal resistance increase
            resistance = vehicle_profile['initial_resistance'] * (1 + (100 - soh) * 0.02)
            
            # Voltage and current simulation
            nominal_voltage = 400 if vehicle_profile['vehicle_type'] in ['bus', 'truck'] else 350
            voltage = nominal_voltage * (soh / 100) + np.random.normal(0, 5)
            current = np.random.uniform(-200, 200)  # Charging/discharging
            
            # Power calculation
            power = voltage * current / 1000  # kW
            
            # Energy consumption
            energy_consumed = abs(power) * (1/60)  # kWh (assuming 1-minute intervals)
            
            data.append({
                'timestamp': current_time,
                'vehicle_id': vehicle_profile['vehicle_id'],
                'vehicle_type': vehicle_profile['vehicle_type'],
                'soh': round(soh, 2),
                'voltage': round(voltage, 2),
                'current': round(current, 2),
                'power': round(power, 2),
                'temperature': round(temperature, 1),
                'humidity': round(humidity, 1),
                'internal_resistance': round(resistance, 4),
                'cycle_count': int(total_cycles),
                'depth_of_discharge': round(depth_of_discharge, 3),
                'energy_consumed': round(energy_consumed, 3),
                'battery_capacity': vehicle_profile['battery_capacity'],
                'age_days': (current_time - start_date).days
            })
        
        return pd.DataFrame(data)
    
    def generate_fleet_data(self, num_vehicles=50, days=365):
        """Generate data for an entire fleet"""
        all_data = []
        
        # Distribute vehicles across types
        vehicle_distribution = {
            'car': int(num_vehicles * 0.4),
            'bus': int(num_vehicles * 0.3),
            'truck': int(num_vehicles * 0.2),
            'motorcycle': int(num_vehicles * 0.1)
        }
        
        vehicle_id = 1
        
        for vehicle_type, count in vehicle_distribution.items():
            for _ in range(count):
                # Generate vehicle profile
                profile = self.generate_vehicle_profile(vehicle_type, f"{vehicle_type}_{vehicle_id:03d}")
                
                # Generate telemetry data
                points_per_day = 24  # Hourly data
                total_points = days * points_per_day
                
                vehicle_data = self.generate_battery_telemetry(profile, total_points)
                all_data.append(vehicle_data)
                
                vehicle_id += 1
        
        # Combine all vehicle data
        fleet_df = pd.concat(all_data, ignore_index=True)
        fleet_df = fleet_df.sort_values('timestamp').reset_index(drop=True)
        
        return fleet_df
    
    def add_data_quality_issues(self, df, missing_rate=0.02, outlier_rate=0.01):
        """Add realistic data quality issues"""
        df_with_issues = df.copy()
        
        # Add missing values
        n_missing = int(len(df) * missing_rate)
        missing_indices = np.random.choice(len(df), n_missing, replace=False)
        
        for idx in missing_indices:
            # Randomly select columns to make missing
            cols_to_miss = np.random.choice(['voltage', 'current', 'temperature'], 
                                          size=np.random.randint(1, 3), replace=False)
            for col in cols_to_miss:
                df_with_issues.loc[idx, col] = np.nan
        
        # Add outliers
        n_outliers = int(len(df) * outlier_rate)
        outlier_indices = np.random.choice(len(df), n_outliers, replace=False)
        
        for idx in outlier_indices:
            # Add voltage outliers
            if np.random.random() < 0.5:
                df_with_issues.loc[idx, 'voltage'] *= np.random.choice([0.5, 2.0])
            
            # Add temperature outliers
            if np.random.random() < 0.5:
                df_with_issues.loc[idx, 'temperature'] += np.random.choice([-50, 50])
        
        return df_with_issues
    
    def get_feature_importance_data(self):
        """Generate feature importance data for ML models"""
        features = [
            'voltage', 'current', 'power', 'temperature', 'humidity',
            'internal_resistance', 'cycle_count', 'depth_of_discharge',
            'age_days', 'energy_consumed'
        ]
        
        # Simulated feature importance scores
        importance_scores = {
            'Random Forest': np.random.dirichlet(np.ones(len(features)) * 2),
            'XGBoost': np.random.dirichlet(np.ones(len(features)) * 2),
            'LSTM': np.random.dirichlet(np.ones(len(features)) * 2)
        }
        
        importance_df = pd.DataFrame(importance_scores, index=features)
        return importance_df

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import SelectKBest, f_regression
import warnings
warnings.filterwarnings('ignore')

class BatteryFeatureEngineer:
    """
    Feature engineering class for battery SoH prediction across different EV types
    """
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_selector = SelectKBest(score_func=f_regression)
        self.engineered_features = []
        
    def create_time_features(self, df):
        """Create time-based features"""
        df = df.copy()
        
        # Ensure timestamp is datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Time-based features
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['month'] = df['timestamp'].dt.month
        df['season'] = df['month'].map({12:0, 1:0, 2:0, 3:1, 4:1, 5:1, 
                                       6:2, 7:2, 8:2, 9:3, 10:3, 11:3})
        
        # Cyclical encoding for time features
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        
        return df
    
    def create_rolling_features(self, df, windows=[7, 30, 90]):
        """Create rolling statistical features"""
        df = df.copy()
        df = df.sort_values(['vehicle_id', 'timestamp'])
        
        numeric_cols = ['voltage', 'current', 'power', 'temperature', 
                       'internal_resistance', 'soh']
        
        for window in windows:
            for col in numeric_cols:
                if col in df.columns:
                    # Rolling mean
                    df[f'{col}_rolling_mean_{window}d'] = (
                        df.groupby('vehicle_id')[col]
                        .rolling(window=window*24, min_periods=1)  # Assuming hourly data
                        .mean().reset_index(0, drop=True)
                    )
                    
                    # Rolling std
                    df[f'{col}_rolling_std_{window}d'] = (
                        df.groupby('vehicle_id')[col]
                        .rolling(window=window*24, min_periods=1)
                        .std().reset_index(0, drop=True)
                    )
                    
                    # Rolling min/max
                    df[f'{col}_rolling_min_{window}d'] = (
                        df.groupby('vehicle_id')[col]
                        .rolling(window=window*24, min_periods=1)
                        .min().reset_index(0, drop=True)
                    )
                    
                    df[f'{col}_rolling_max_{window}d'] = (
                        df.groupby('vehicle_id')[col]
                        .rolling(window=window*24, min_periods=1)
                        .max().reset_index(0, drop=True)
                    )
        
        return df
    
    def create_degradation_features(self, df):
        """Create battery degradation-specific features"""
        df = df.copy()
        df = df.sort_values(['vehicle_id', 'timestamp'])
        
        # Rate of change features
        numeric_cols = ['soh', 'internal_resistance', 'voltage']
        
        for col in numeric_cols:
            if col in df.columns:
                df[f'{col}_rate_of_change'] = (
                    df.groupby('vehicle_id')[col]
                    .pct_change(periods=24)  # Daily change rate
                    .fillna(0)
                )
                
                # Cumulative change from first measurement
                df[f'{col}_cumulative_change'] = (
                    df.groupby('vehicle_id')[col]
                    .apply(lambda x: (x - x.iloc[0]) / x.iloc[0] if x.iloc[0] != 0 else 0)
                    .reset_index(0, drop=True)
                )
        
        # Efficiency metrics
        df['energy_efficiency'] = np.where(
            df['energy_consumed'] > 0,
            df['power'] / df['energy_consumed'],
            0
        )
        
        # Temperature stress indicator
        df['temp_stress'] = np.where(
            (df['temperature'] < 0) | (df['temperature'] > 40),
            1, 0
        )
        
        # Cycle stress (high DoD cycles)
        df['cycle_stress'] = np.where(
            df['depth_of_discharge'] > 0.8,
            1, 0
        )
        
        return df
    
    def create_vehicle_specific_features(self, df):
        """Create features specific to vehicle types"""
        df = df.copy()
        
        # Vehicle type encoding
        if 'vehicle_type' in df.columns:
            df['vehicle_type_encoded'] = self.label_encoder.fit_transform(df['vehicle_type'])
        
        # Normalized features by vehicle type
        vehicle_specific_cols = ['battery_capacity', 'cycle_count', 'age_days']
        
        for col in vehicle_specific_cols:
            if col in df.columns:
                # Normalize by vehicle type percentiles
                df[f'{col}_normalized'] = (
                    df.groupby('vehicle_type')[col]
                    .rank(pct=True)
                )
                
                # Z-score within vehicle type
                df[f'{col}_zscore'] = (
                    df.groupby('vehicle_type')[col]
                    .transform(lambda x: (x - x.mean()) / x.std() if x.std() > 0 else 0)
                )
        
        # Power density (power per unit capacity)
        df['power_density'] = df['power'] / df['battery_capacity']
        
        # Capacity utilization
        df['capacity_utilization'] = df['energy_consumed'] / df['battery_capacity']
        
        return df
    
    def create_interaction_features(self, df):
        """Create interaction features between important variables"""
        df = df.copy()
        
        # Temperature-power interaction
        df['temp_power_interaction'] = df['temperature'] * df['power']
        
        # Age-cycle interaction
        df['age_cycle_interaction'] = df['age_days'] * df['cycle_count']
        
        # DoD-temperature interaction
        df['dod_temp_interaction'] = df['depth_of_discharge'] * df['temperature']
        
        # Resistance-voltage ratio
        df['resistance_voltage_ratio'] = np.where(
            df['voltage'] > 0,
            df['internal_resistance'] / df['voltage'],
            0
        )
        
        # Power efficiency
        df['power_efficiency'] = np.where(
            (df['voltage'] > 0) & (df['current'] != 0),
            df['power'] / (df['voltage'] * abs(df['current'])),
            0
        )
        
        return df
    
    def create_lag_features(self, df, lags=[1, 6, 24]):
        """Create lag features for time series analysis"""
        df = df.copy()
        df = df.sort_values(['vehicle_id', 'timestamp'])
        
        lag_cols = ['soh', 'voltage', 'current', 'temperature', 'internal_resistance']
        
        for lag in lags:
            for col in lag_cols:
                if col in df.columns:
                    df[f'{col}_lag_{lag}'] = (
                        df.groupby('vehicle_id')[col]
                        .shift(lag)
                        .fillna(method='bfill')
                    )
        
        return df
    
    def engineer_all_features(self, df):
        """Apply all feature engineering steps"""
        print("Starting feature engineering...")
        
        # Create base features
        df_engineered = self.create_time_features(df)
        print("✓ Time features created")
        
        df_engineered = self.create_degradation_features(df_engineered)
        print("✓ Degradation features created")
        
        df_engineered = self.create_vehicle_specific_features(df_engineered)
        print("✓ Vehicle-specific features created")
        
        df_engineered = self.create_interaction_features(df_engineered)
        print("✓ Interaction features created")
        
        df_engineered = self.create_lag_features(df_engineered)
        print("✓ Lag features created")
        
        # Create rolling features (this can be computationally expensive)
        print("Creating rolling features (this may take a moment)...")
        df_engineered = self.create_rolling_features(df_engineered, windows=[7, 30])
        print("✓ Rolling features created")
        
        # Store engineered feature names
        self.engineered_features = [col for col in df_engineered.columns 
                                   if col not in df.columns]
        
        print(f"Feature engineering complete. Added {len(self.engineered_features)} new features.")
        
        return df_engineered
    
    def select_features(self, X, y, k=50):
        """Select top k features using statistical tests"""
        if len(X.columns) <= k:
            return X
        
        # Fill NaN values for feature selection
        X_filled = X.fillna(X.mean())
        
        # Select features
        X_selected = self.feature_selector.fit_transform(X_filled, y)
        selected_features = X.columns[self.feature_selector.get_support()]
        
        print(f"Selected {len(selected_features)} features out of {len(X.columns)}")
        
        return X[selected_features]
    
    def prepare_features_for_ml(self, df, target_col='soh', test_size=0.2):
        """Prepare features for machine learning"""
        # Remove non-feature columns
        feature_cols = [col for col in df.columns if col not in [
            'timestamp', 'vehicle_id', target_col
        ]]
        
        X = df[feature_cols].copy()
        y = df[target_col].copy()
        
        # Handle missing values
        X = X.fillna(X.mean())
        
        # Remove infinite values
        X = X.replace([np.inf, -np.inf], np.nan).fillna(X.mean())
        
        # Scale features
        X_scaled = pd.DataFrame(
            self.scaler.fit_transform(X),
            columns=X.columns,
            index=X.index
        )
        
        return X_scaled, y
    
    def get_feature_importance_summary(self, df):
        """Get summary of engineered features"""
        summary = {
            'Total Features': len(df.columns),
            'Original Features': len([col for col in df.columns 
                                    if col not in self.engineered_features]),
            'Engineered Features': len(self.engineered_features),
            'Feature Categories': {
                'Time Features': len([col for col in self.engineered_features 
                                    if any(x in col for x in ['hour', 'day', 'month', 'season'])]),
                'Rolling Features': len([col for col in self.engineered_features 
                                       if 'rolling' in col]),
                'Degradation Features': len([col for col in self.engineered_features 
                                           if any(x in col for x in ['rate_of_change', 'cumulative'])]),
                'Interaction Features': len([col for col in self.engineered_features 
                                           if 'interaction' in col or 'ratio' in col]),
                'Lag Features': len([col for col in self.engineered_features 
                                   if 'lag' in col])
            }
        }
        
        return summary

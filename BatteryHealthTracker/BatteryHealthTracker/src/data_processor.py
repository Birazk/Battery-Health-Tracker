import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings('ignore')

class BatteryDataProcessor:
    """
    Data processing utilities for battery telemetry data
    """
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.imputer = SimpleImputer(strategy='median')
        self.data_quality_report = {}
        
    def validate_data_quality(self, df):
        """Validate and report data quality issues"""
        quality_report = {
            'total_records': len(df),
            'missing_values': {},
            'outliers': {},
            'data_types': {},
            'duplicates': df.duplicated().sum(),
            'time_gaps': 0
        }
        
        # Check missing values
        for col in df.columns:
            missing_count = df[col].isnull().sum()
            missing_pct = (missing_count / len(df)) * 100
            quality_report['missing_values'][col] = {
                'count': missing_count,
                'percentage': round(missing_pct, 2)
            }
        
        # Check data types
        for col in df.columns:
            quality_report['data_types'][col] = str(df[col].dtype)
        
        # Check for outliers in numerical columns
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        for col in numerical_cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
            quality_report['outliers'][col] = {
                'count': len(outliers),
                'percentage': round((len(outliers) / len(df)) * 100, 2)
            }
        
        # Check time gaps (if timestamp column exists)
        if 'timestamp' in df.columns:
            df_sorted = df.sort_values('timestamp')
            time_diffs = df_sorted['timestamp'].diff()
            expected_interval = time_diffs.median()
            large_gaps = time_diffs > (expected_interval * 2)
            quality_report['time_gaps'] = large_gaps.sum()
        
        self.data_quality_report = quality_report
        return quality_report
    
    def clean_data(self, df):
        """Clean and preprocess the data"""
        df_clean = df.copy()
        
        # Remove duplicates
        df_clean = df_clean.drop_duplicates()
        
        # Handle missing values
        numerical_cols = df_clean.select_dtypes(include=[np.number]).columns
        categorical_cols = df_clean.select_dtypes(include=['object']).columns
        
        # Fill numerical missing values with median
        for col in numerical_cols:
            if df_clean[col].isnull().sum() > 0:
                median_val = df_clean[col].median()
                df_clean[col] = df_clean[col].fillna(median_val)
        
        # Fill categorical missing values with mode
        for col in categorical_cols:
            if col != 'timestamp' and df_clean[col].isnull().sum() > 0:
                mode_val = df_clean[col].mode()
                if not mode_val.empty:
                    df_clean[col] = df_clean[col].fillna(mode_val[0])
        
        # Handle outliers (cap extreme values)
        for col in numerical_cols:
            if col in ['soh', 'voltage', 'current', 'power', 'temperature']:
                Q1 = df_clean[col].quantile(0.01)
                Q99 = df_clean[col].quantile(0.99)
                df_clean[col] = df_clean[col].clip(lower=Q1, upper=Q99)
        
        # Ensure SoH is within valid range
        if 'soh' in df_clean.columns:
            df_clean['soh'] = df_clean['soh'].clip(lower=70, upper=100)
        
        # Ensure timestamp is datetime
        if 'timestamp' in df_clean.columns:
            df_clean['timestamp'] = pd.to_datetime(df_clean['timestamp'])
        
        return df_clean
    
    def detect_anomalies(self, df, vehicle_id=None):
        """Detect anomalies in battery data"""
        anomalies = []
        
        # Filter for specific vehicle if provided
        if vehicle_id:
            df_vehicle = df[df['vehicle_id'] == vehicle_id].copy()
        else:
            df_vehicle = df.copy()
        
        # Sort by timestamp
        df_vehicle = df_vehicle.sort_values('timestamp')
        
        # 1. Sudden SoH drops
        if 'soh' in df_vehicle.columns:
            df_vehicle['soh_change'] = df_vehicle['soh'].diff()
            sudden_drops = df_vehicle[df_vehicle['soh_change'] < -5]  # >5% sudden drop
            
            for _, row in sudden_drops.iterrows():
                anomalies.append({
                    'type': 'Sudden SoH Drop',
                    'timestamp': row['timestamp'],
                    'vehicle_id': row['vehicle_id'],
                    'value': row['soh_change'],
                    'severity': 'High' if row['soh_change'] < -10 else 'Medium'
                })
        
        # 2. Temperature extremes
        if 'temperature' in df_vehicle.columns:
            temp_extremes = df_vehicle[
                (df_vehicle['temperature'] < -20) | 
                (df_vehicle['temperature'] > 60)
            ]
            
            for _, row in temp_extremes.iterrows():
                anomalies.append({
                    'type': 'Temperature Extreme',
                    'timestamp': row['timestamp'],
                    'vehicle_id': row['vehicle_id'],
                    'value': row['temperature'],
                    'severity': 'High' if abs(row['temperature']) > 50 else 'Medium'
                })
        
        # 3. Voltage anomalies
        if 'voltage' in df_vehicle.columns:
            voltage_mean = df_vehicle['voltage'].mean()
            voltage_std = df_vehicle['voltage'].std()
            voltage_anomalies = df_vehicle[
                abs(df_vehicle['voltage'] - voltage_mean) > (3 * voltage_std)
            ]
            
            for _, row in voltage_anomalies.iterrows():
                anomalies.append({
                    'type': 'Voltage Anomaly',
                    'timestamp': row['timestamp'],
                    'vehicle_id': row['vehicle_id'],
                    'value': row['voltage'],
                    'severity': 'Medium'
                })
        
        # 4. Power inconsistencies
        if all(col in df_vehicle.columns for col in ['voltage', 'current', 'power']):
            df_vehicle['calculated_power'] = df_vehicle['voltage'] * df_vehicle['current'] / 1000
            df_vehicle['power_diff'] = abs(df_vehicle['power'] - df_vehicle['calculated_power'])
            
            power_inconsistencies = df_vehicle[df_vehicle['power_diff'] > 10]  # >10kW difference
            
            for _, row in power_inconsistencies.iterrows():
                anomalies.append({
                    'type': 'Power Inconsistency',
                    'timestamp': row['timestamp'],
                    'vehicle_id': row['vehicle_id'],
                    'value': row['power_diff'],
                    'severity': 'Low'
                })
        
        return pd.DataFrame(anomalies)
    
    def aggregate_data(self, df, freq='H', vehicle_id=None):
        """Aggregate data by time frequency"""
        if vehicle_id:
            df_agg = df[df['vehicle_id'] == vehicle_id].copy()
        else:
            df_agg = df.copy()
        
        # Ensure timestamp is datetime and set as index
        df_agg['timestamp'] = pd.to_datetime(df_agg['timestamp'])
        df_agg = df_agg.set_index('timestamp')
        
        # Define aggregation functions for different columns
        agg_functions = {
            'soh': 'mean',
            'voltage': 'mean',
            'current': 'mean',
            'power': 'mean',
            'temperature': 'mean',
            'humidity': 'mean',
            'internal_resistance': 'mean',
            'cycle_count': 'last',
            'depth_of_discharge': 'mean',
            'energy_consumed': 'sum',
            'battery_capacity': 'first',
            'age_days': 'last'
        }
        
        # Select only columns that exist in the dataframe
        available_agg = {k: v for k, v in agg_functions.items() if k in df_agg.columns}
        
        # Group by vehicle and resample
        if 'vehicle_id' in df_agg.columns:
            df_resampled = (df_agg.groupby('vehicle_id')
                           .resample(freq)
                           .agg(available_agg)
                           .reset_index())
        else:
            df_resampled = df_agg.resample(freq).agg(available_agg).reset_index()
        
        return df_resampled
    
    def calculate_degradation_metrics(self, df):
        """Calculate battery degradation metrics"""
        metrics = {}
        
        for vehicle_id in df['vehicle_id'].unique():
            vehicle_data = df[df['vehicle_id'] == vehicle_id].sort_values('timestamp')
            
            if len(vehicle_data) < 2:
                continue
            
            # Initial and current SoH
            initial_soh = vehicle_data['soh'].iloc[0]
            current_soh = vehicle_data['soh'].iloc[-1]
            
            # Time span
            time_span_days = (vehicle_data['timestamp'].iloc[-1] - 
                            vehicle_data['timestamp'].iloc[0]).days
            
            # Degradation rate
            total_degradation = initial_soh - current_soh
            degradation_rate_per_year = (total_degradation / time_span_days) * 365 if time_span_days > 0 else 0
            
            # Linear degradation trend
            x = np.arange(len(vehicle_data))
            y = vehicle_data['soh'].values
            if len(x) > 1:
                slope = np.polyfit(x, y, 1)[0]
            else:
                slope = 0
            
            # Remaining useful life estimation (assuming 80% SoH as end of life)
            if degradation_rate_per_year > 0:
                remaining_capacity = current_soh - 80
                rul_years = remaining_capacity / degradation_rate_per_year
            else:
                rul_years = float('inf')
            
            metrics[vehicle_id] = {
                'vehicle_type': vehicle_data['vehicle_type'].iloc[-1],
                'initial_soh': round(initial_soh, 2),
                'current_soh': round(current_soh, 2),
                'total_degradation': round(total_degradation, 2),
                'degradation_rate_per_year': round(degradation_rate_per_year, 2),
                'degradation_trend_slope': round(slope, 4),
                'time_span_days': time_span_days,
                'estimated_rul_years': round(rul_years, 1) if rul_years != float('inf') else '>10',
                'health_status': self._get_health_status(current_soh)
            }
        
        return pd.DataFrame.from_dict(metrics, orient='index').reset_index()
    
    def _get_health_status(self, soh):
        """Determine health status based on SoH"""
        if soh >= 95:
            return 'Excellent'
        elif soh >= 90:
            return 'Good'
        elif soh >= 85:
            return 'Fair'
        elif soh >= 80:
            return 'Poor'
        else:
            return 'Critical'
    
    def get_data_quality_summary(self):
        """Get summary of data quality report"""
        if not self.data_quality_report:
            return "No data quality report available. Run validate_data_quality() first."
        
        report = self.data_quality_report
        
        # Count columns with significant missing values
        high_missing = sum(1 for col_info in report['missing_values'].values() 
                          if col_info['percentage'] > 10)
        
        # Count columns with many outliers
        high_outliers = sum(1 for col_info in report['outliers'].values() 
                           if col_info['percentage'] > 5)
        
        summary = {
            'total_records': report['total_records'],
            'duplicate_records': report['duplicates'],
            'columns_with_high_missing_values': high_missing,
            'columns_with_many_outliers': high_outliers,
            'time_gaps_detected': report['time_gaps'],
            'overall_quality': self._assess_overall_quality(report)
        }
        
        return summary
    
    def _assess_overall_quality(self, report):
        """Assess overall data quality"""
        issues = 0
        
        # Check for high missing values
        if any(col_info['percentage'] > 20 for col_info in report['missing_values'].values()):
            issues += 2
        elif any(col_info['percentage'] > 10 for col_info in report['missing_values'].values()):
            issues += 1
        
        # Check for duplicates
        if report['duplicates'] > len(report) * 0.05:  # >5% duplicates
            issues += 1
        
        # Check for outliers
        if any(col_info['percentage'] > 10 for col_info in report['outliers'].values()):
            issues += 1
        
        # Check for time gaps
        if report['time_gaps'] > 10:
            issues += 1
        
        if issues == 0:
            return 'Excellent'
        elif issues == 1:
            return 'Good'
        elif issues == 2:
            return 'Fair'
        else:
            return 'Poor'

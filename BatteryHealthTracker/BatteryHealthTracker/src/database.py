import os
import pandas as pd
import numpy as np
from sqlalchemy import create_engine, MetaData, Table, Column, Integer, String, Float, DateTime, Text, Boolean
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.dialects.postgresql import UUID
import uuid
from datetime import datetime
import pickle
import base64
import warnings
warnings.filterwarnings('ignore')

Base = declarative_base()

class BatteryTelemetryData(Base):
    __tablename__ = 'battery_telemetry'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    vehicle_id = Column(String(50), nullable=False, index=True)
    vehicle_type = Column(String(20), nullable=False, index=True)
    timestamp = Column(DateTime, nullable=False, index=True)
    soh = Column(Float, nullable=False)
    voltage = Column(Float, nullable=False)
    current = Column(Float, nullable=False)
    power = Column(Float, nullable=False)
    temperature = Column(Float, nullable=False)
    humidity = Column(Float)
    internal_resistance = Column(Float, nullable=False)
    cycle_count = Column(Integer, nullable=False)
    depth_of_discharge = Column(Float, nullable=False)
    energy_consumed = Column(Float, nullable=False)
    battery_capacity = Column(Float, nullable=False)
    age_days = Column(Integer, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

class VehicleProfile(Base):
    __tablename__ = 'vehicle_profiles'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    vehicle_id = Column(String(50), unique=True, nullable=False, index=True)
    vehicle_type = Column(String(20), nullable=False)
    battery_capacity = Column(Float, nullable=False)
    degradation_rate = Column(Float, nullable=False)
    usage_pattern = Column(String(50), nullable=False)
    temperature_sensitivity = Column(Float, nullable=False)
    manufacturing_date = Column(DateTime, nullable=False)
    initial_resistance = Column(Float, nullable=False)
    cycle_life_expected = Column(Integer, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

class ModelResults(Base):
    __tablename__ = 'model_results'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    model_name = Column(String(50), nullable=False)
    model_type = Column(String(50), nullable=False)
    accuracy_percentage = Column(Float, nullable=False)
    test_r2 = Column(Float, nullable=False)
    train_r2 = Column(Float, nullable=False)
    mae = Column(Float, nullable=False)
    rmse = Column(Float, nullable=False)
    model_data_blob = Column(Text)  # Serialized model data
    feature_importance_blob = Column(Text)  # Serialized feature importance
    training_features = Column(Text)  # JSON list of features
    created_at = Column(DateTime, default=datetime.utcnow)
    is_active = Column(Boolean, default=True)

class PredictionHistory(Base):
    __tablename__ = 'prediction_history'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    vehicle_id = Column(String(50), nullable=False, index=True)
    model_name = Column(String(50), nullable=False)
    predicted_soh = Column(Float, nullable=False)
    actual_soh = Column(Float)
    confidence_lower = Column(Float)
    confidence_upper = Column(Float)
    prediction_timestamp = Column(DateTime, default=datetime.utcnow)
    input_features_blob = Column(Text)  # Serialized input features

class DatabaseManager:
    """
    Database manager for EV Battery SoH Prognostics System
    """
    
    def __init__(self):
        self.database_url = os.getenv('DATABASE_URL')
        if not self.database_url:
            raise ValueError("DATABASE_URL environment variable not found")
        
        self.engine = create_engine(self.database_url)
        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
        self.metadata = MetaData()
        
        # Create tables
        self.create_tables()
    
    def create_tables(self):
        """Create all database tables"""
        try:
            Base.metadata.create_all(bind=self.engine)
            print("✓ Database tables created successfully")
        except Exception as e:
            print(f"Error creating database tables: {str(e)}")
            raise
    
    def get_session(self):
        """Get database session"""
        return self.SessionLocal()
    
    def save_fleet_data(self, fleet_df):
        """Save fleet telemetry data to database"""
        session = self.get_session()
        try:
            # Clear existing data for clean insert
            session.query(BatteryTelemetryData).delete()
            session.commit()
            
            # Convert DataFrame to database records
            records = []
            for _, row in fleet_df.iterrows():
                record = BatteryTelemetryData(
                    vehicle_id=row['vehicle_id'],
                    vehicle_type=row['vehicle_type'],
                    timestamp=pd.to_datetime(row['timestamp']),
                    soh=float(row['soh']),
                    voltage=float(row['voltage']),
                    current=float(row['current']),
                    power=float(row['power']),
                    temperature=float(row['temperature']),
                    humidity=float(row.get('humidity', 0)),
                    internal_resistance=float(row['internal_resistance']),
                    cycle_count=int(row['cycle_count']),
                    depth_of_discharge=float(row['depth_of_discharge']),
                    energy_consumed=float(row['energy_consumed']),
                    battery_capacity=float(row['battery_capacity']),
                    age_days=int(row['age_days']),
                )
                records.append(record)
            
            # Batch insert
            session.bulk_save_objects(records)
            session.commit()
            
            print(f"✓ Saved {len(records)} telemetry records to database")
            return True
            
        except Exception as e:
            session.rollback()
            print(f"Error saving fleet data: {str(e)}")
            return False
        finally:
            session.close()
    
    def load_fleet_data(self):
        """Load fleet telemetry data from database"""
        session = self.get_session()
        try:
            # Query all telemetry data
            records = session.query(BatteryTelemetryData).all()
            
            if not records:
                return None
            
            # Convert to DataFrame
            data = []
            for record in records:
                data.append({
                    'vehicle_id': record.vehicle_id,
                    'vehicle_type': record.vehicle_type,
                    'timestamp': record.timestamp,
                    'soh': record.soh,
                    'voltage': record.voltage,
                    'current': record.current,
                    'power': record.power,
                    'temperature': record.temperature,
                    'humidity': record.humidity,
                    'internal_resistance': record.internal_resistance,
                    'cycle_count': record.cycle_count,
                    'depth_of_discharge': record.depth_of_discharge,
                    'energy_consumed': record.energy_consumed,
                    'battery_capacity': record.battery_capacity,
                    'age_days': record.age_days
                })
            
            df = pd.DataFrame(data)
            print(f"✓ Loaded {len(df)} telemetry records from database")
            return df
            
        except Exception as e:
            print(f"Error loading fleet data: {str(e)}")
            return None
        finally:
            session.close()
    
    def save_vehicle_profiles(self, profiles_list):
        """Save vehicle profiles to database"""
        session = self.get_session()
        try:
            # Clear existing profiles
            session.query(VehicleProfile).delete()
            session.commit()
            
            records = []
            for profile in profiles_list:
                record = VehicleProfile(
                    vehicle_id=profile['vehicle_id'],
                    vehicle_type=profile['vehicle_type'],
                    battery_capacity=profile['battery_capacity'],
                    degradation_rate=profile['degradation_rate'],
                    usage_pattern=profile['usage_pattern'],
                    temperature_sensitivity=profile['temperature_sensitivity'],
                    manufacturing_date=profile['manufacturing_date'],
                    initial_resistance=profile['initial_resistance'],
                    cycle_life_expected=profile['cycle_life_expected']
                )
                records.append(record)
            
            session.bulk_save_objects(records)
            session.commit()
            
            print(f"✓ Saved {len(records)} vehicle profiles to database")
            return True
            
        except Exception as e:
            session.rollback()
            print(f"Error saving vehicle profiles: {str(e)}")
            return False
        finally:
            session.close()
    
    def save_model_results(self, ml_models_obj):
        """Save trained model results to database"""
        session = self.get_session()
        try:
            # Mark existing models as inactive
            session.query(ModelResults).update({'is_active': False})
            
            # Save new model results
            for model_name, scores in ml_models_obj.model_scores.items():
                # Serialize model data (simplified - in production use proper model serialization)
                model_data_blob = None
                feature_importance_blob = None
                
                if hasattr(ml_models_obj, 'feature_importance') and model_name in ml_models_obj.feature_importance:
                    feature_importance_df = ml_models_obj.feature_importance[model_name]
                    feature_importance_blob = base64.b64encode(
                        pickle.dumps(feature_importance_df.to_dict())
                    ).decode('utf-8')
                
                # Get training features
                training_features = None
                if hasattr(ml_models_obj, 'training_features'):
                    training_features = str(ml_models_obj.training_features)
                
                record = ModelResults(
                    model_name=model_name,
                    model_type=model_name.lower().replace(' ', '_'),
                    accuracy_percentage=scores['accuracy_percentage'],
                    test_r2=scores['test_r2'],
                    train_r2=scores['train_r2'],
                    mae=scores['mae'],
                    rmse=scores['rmse'],
                    model_data_blob=model_data_blob,
                    feature_importance_blob=feature_importance_blob,
                    training_features=training_features,
                    is_active=True
                )
                
                session.add(record)
            
            session.commit()
            print(f"✓ Saved {len(ml_models_obj.model_scores)} model results to database")
            return True
            
        except Exception as e:
            session.rollback()
            print(f"Error saving model results: {str(e)}")
            return False
        finally:
            session.close()
    
    def load_model_results(self):
        """Load the latest model results from database"""
        session = self.get_session()
        try:
            records = session.query(ModelResults).filter(ModelResults.is_active == True).all()
            
            if not records:
                return None
            
            model_scores = {}
            feature_importance = {}
            
            for record in records:
                model_scores[record.model_name] = {
                    'accuracy_percentage': record.accuracy_percentage,
                    'test_r2': record.test_r2,
                    'train_r2': record.train_r2,
                    'mae': record.mae,
                    'rmse': record.rmse
                }
                
                # Deserialize feature importance if available
                if record.feature_importance_blob:
                    try:
                        importance_dict = pickle.loads(
                            base64.b64decode(record.feature_importance_blob.encode('utf-8'))
                        )
                        feature_importance[record.model_name] = pd.DataFrame(importance_dict)
                    except Exception as e:
                        print(f"Warning: Could not load feature importance for {record.model_name}: {e}")
            
            print(f"✓ Loaded {len(model_scores)} model results from database")
            return {
                'model_scores': model_scores,
                'feature_importance': feature_importance
            }
            
        except Exception as e:
            print(f"Error loading model results: {str(e)}")
            return None
        finally:
            session.close()
    
    def save_prediction(self, vehicle_id, model_name, predicted_soh, actual_soh=None, 
                       confidence_interval=None, input_features=None):
        """Save a prediction to the database"""
        session = self.get_session()
        try:
            # Serialize input features
            input_features_blob = None
            if input_features is not None:
                input_features_blob = base64.b64encode(
                    pickle.dumps(input_features.to_dict() if hasattr(input_features, 'to_dict') else input_features)
                ).decode('utf-8')
            
            confidence_lower = None
            confidence_upper = None
            if confidence_interval:
                confidence_lower = confidence_interval[0]
                confidence_upper = confidence_interval[1]
            
            record = PredictionHistory(
                vehicle_id=vehicle_id,
                model_name=model_name,
                predicted_soh=predicted_soh,
                actual_soh=actual_soh,
                confidence_lower=confidence_lower,
                confidence_upper=confidence_upper,
                input_features_blob=input_features_blob
            )
            
            session.add(record)
            session.commit()
            return True
            
        except Exception as e:
            session.rollback()
            print(f"Error saving prediction: {str(e)}")
            return False
        finally:
            session.close()
    
    def get_prediction_history(self, vehicle_id=None, model_name=None, limit=100):
        """Get prediction history from database"""
        session = self.get_session()
        try:
            query = session.query(PredictionHistory)
            
            if vehicle_id:
                query = query.filter(PredictionHistory.vehicle_id == vehicle_id)
            
            if model_name:
                query = query.filter(PredictionHistory.model_name == model_name)
            
            records = query.order_by(PredictionHistory.prediction_timestamp.desc()).limit(limit).all()
            
            predictions = []
            for record in records:
                predictions.append({
                    'vehicle_id': record.vehicle_id,
                    'model_name': record.model_name,
                    'predicted_soh': record.predicted_soh,
                    'actual_soh': record.actual_soh,
                    'confidence_lower': record.confidence_lower,
                    'confidence_upper': record.confidence_upper,
                    'timestamp': record.prediction_timestamp
                })
            
            return pd.DataFrame(predictions)
            
        except Exception as e:
            print(f"Error getting prediction history: {str(e)}")
            return pd.DataFrame()
        finally:
            session.close()
    
    def get_database_stats(self):
        """Get database statistics"""
        session = self.get_session()
        try:
            stats = {
                'telemetry_records': session.query(BatteryTelemetryData).count(),
                'vehicle_profiles': session.query(VehicleProfile).count(),
                'model_results': session.query(ModelResults).filter(ModelResults.is_active == True).count(),
                'prediction_history': session.query(PredictionHistory).count(),
                'unique_vehicles': session.query(BatteryTelemetryData.vehicle_id).distinct().count()
            }
            
            return stats
            
        except Exception as e:
            print(f"Error getting database stats: {str(e)}")
            return {}
        finally:
            session.close()
    
    def cleanup_old_data(self, days_to_keep=90):
        """Clean up old data to maintain database performance"""
        session = self.get_session()
        try:
            from datetime import timedelta
            cutoff_date = datetime.utcnow() - timedelta(days=days_to_keep)
            
            # Clean up old predictions
            deleted_predictions = session.query(PredictionHistory).filter(
                PredictionHistory.prediction_timestamp < cutoff_date
            ).delete()
            
            # Clean up inactive model results
            deleted_models = session.query(ModelResults).filter(
                ModelResults.is_active == False,
                ModelResults.created_at < cutoff_date
            ).delete()
            
            session.commit()
            
            print(f"✓ Cleaned up {deleted_predictions} old predictions and {deleted_models} old model results")
            return True
            
        except Exception as e:
            session.rollback()
            print(f"Error during cleanup: {str(e)}")
            return False
        finally:
            session.close()
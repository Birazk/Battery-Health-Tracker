import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from src.database import DatabaseManager
from datetime import datetime, timedelta
import time

# Page configuration
st.set_page_config(page_title="Real-Time Prediction", page_icon="🔮", layout="wide")

st.title("🔮 Real-Time State-of-Health Prediction")
st.markdown("### Live battery degradation forecasting with confidence intervals")

# Check for trained models
if 'ml_models' not in st.session_state or not st.session_state.ml_models.is_trained:
    st.error("❌ No trained models available. Please train models in the 'Model Training' tab first.")
    st.stop()

if 'fleet_data' not in st.session_state:
    st.error("❌ No fleet data available. Please generate data in the 'Data Ingestion' tab first.")
    st.stop()

ml_models = st.session_state.ml_models
fleet_data = st.session_state.fleet_data

# Initialize database manager
if 'db_manager' not in st.session_state:
    try:
        st.session_state.db_manager = DatabaseManager()
    except Exception as e:
        st.error(f"Database connection failed: {str(e)}")
        st.session_state.db_manager = None

# Sidebar for prediction settings
with st.sidebar:
    st.header("Prediction Settings")
    
    # Model selection
    available_models = list(ml_models.model_scores.keys())
    selected_model = st.selectbox("Select Prediction Model", available_models)
    
    # Vehicle selection
    st.subheader("Vehicle Selection")
    vehicle_ids = fleet_data['vehicle_id'].unique()
    selected_vehicle = st.selectbox("Select Vehicle", vehicle_ids)
    
    # Prediction parameters
    st.subheader("Prediction Parameters")
    prediction_horizon = st.slider("Prediction Horizon (days)", 1, 365, 30)
    confidence_level = st.slider("Confidence Level (%)", 80, 99, 95)
    
    # Real-time simulation
    st.subheader("Real-Time Simulation")
    enable_realtime = st.checkbox("Enable Real-Time Updates", False)
    if enable_realtime:
        update_interval = st.slider("Update Interval (seconds)", 1, 10, 3)
    
    # Alert settings
    st.subheader("Alert Settings")
    enable_alerts = st.checkbox("Enable Alerts", True)
    if enable_alerts:
        warning_threshold = st.slider("Warning Threshold (%)", 70, 95, 85)
        critical_threshold = st.slider("Critical Threshold (%)", 70, 90, 80)

# Main content tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "Live Prediction", 
    "Trend Analysis", 
    "Vehicle Diagnostics", 
    "Prediction History"
])

def get_latest_vehicle_data(vehicle_id):
    """Get the latest data for a specific vehicle"""
    vehicle_data = fleet_data[fleet_data['vehicle_id'] == vehicle_id].copy()
    if vehicle_data.empty:
        return None
    
    # Sort by timestamp and get latest
    vehicle_data = vehicle_data.sort_values('timestamp')
    latest_data = vehicle_data.iloc[-1]
    
    return latest_data

def make_prediction_with_confidence(vehicle_data, model_name):
    """Make prediction with confidence intervals"""
    try:
        # Prepare data for prediction
        if 'engineered_data' in st.session_state and 'training_features' in st.session_state:
            # Use engineered features if available
            engineered_data = st.session_state.engineered_data
            training_features = st.session_state.training_features
            
            # Find matching row in engineered data
            vehicle_engineered = engineered_data[
                engineered_data['vehicle_id'] == vehicle_data['vehicle_id']
            ].iloc[-1]  # Get latest
            
            # Select only training features
            prediction_data = vehicle_engineered[training_features]
            
        else:
            # Use basic features
            feature_cols = ['voltage', 'current', 'power', 'temperature', 
                          'internal_resistance', 'cycle_count', 'age_days']
            available_features = [col for col in feature_cols if col in vehicle_data.index]
            prediction_data = vehicle_data[available_features]
        
        # Make prediction
        result = ml_models.predict_single_vehicle(
            prediction_data, 
            model_name=model_name
        )
        
        return result
        
    except Exception as e:
        st.error(f"Prediction failed: {str(e)}")
        return None

def generate_degradation_trend(current_soh, days_ahead, degradation_rate=0.01):
    """Generate future degradation trend"""
    dates = [datetime.now() + timedelta(days=i) for i in range(days_ahead + 1)]
    
    # Base degradation with some variability
    base_trend = []
    for i in range(days_ahead + 1):
        # Add some realistic variability
        daily_variation = np.random.normal(0, 0.1)
        predicted_soh = max(70, current_soh - (degradation_rate * i) + daily_variation)
        base_trend.append(predicted_soh)
    
    # Confidence intervals
    confidence_width = 2.0  # ±2% confidence interval
    upper_bound = [min(100, soh + confidence_width) for soh in base_trend]
    lower_bound = [max(70, soh - confidence_width) for soh in base_trend]
    
    return dates, base_trend, upper_bound, lower_bound

with tab1:
    st.header("Live State-of-Health Prediction")
    
    # Get latest vehicle data
    latest_data = get_latest_vehicle_data(selected_vehicle)
    
    if latest_data is None:
        st.error(f"No data available for vehicle {selected_vehicle}")
        st.stop()
    
    # Create columns for layout
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        st.subheader(f"Vehicle: {selected_vehicle}")
        
        # Vehicle information
        vehicle_info = {
            "Vehicle Type": latest_data['vehicle_type'],
            "Last Update": latest_data['timestamp'].strftime("%Y-%m-%d %H:%M:%S"),
            "Battery Capacity": f"{latest_data['battery_capacity']:.1f} kWh",
            "Age": f"{latest_data['age_days']} days",
            "Cycle Count": f"{latest_data['cycle_count']:,.0f}"
        }
        
        for key, value in vehicle_info.items():
            st.write(f"**{key}:** {value}")
    
    with col2:
        # Real-time prediction
        if st.button("🔮 Predict SoH", type="primary") or enable_realtime:
            prediction_result = make_prediction_with_confidence(latest_data, selected_model)
            
            if prediction_result:
                # Save prediction to database
                if st.session_state.db_manager:
                    st.session_state.db_manager.save_prediction(
                        vehicle_id=selected_vehicle,
                        model_name=selected_model,
                        predicted_soh=prediction_result['prediction'],
                        actual_soh=latest_data['soh'],
                        confidence_interval=(prediction_result.get('confidence_lower'), 
                                           prediction_result.get('confidence_upper')),
                        input_features=latest_data.to_dict()
                    )
                predicted_soh = prediction_result['prediction']
                confidence_interval = prediction_result['confidence_interval']
                model_accuracy = prediction_result['model_accuracy']
                
                # Store prediction in session state
                st.session_state.latest_prediction = prediction_result
                st.session_state.prediction_timestamp = datetime.now()
                
                # Display prediction
                delta_color = "normal" if predicted_soh > warning_threshold else "inverse"
                st.metric(
                    "Predicted SoH",
                    f"{predicted_soh:.1f}%",
                    delta=f"±{(confidence_interval[1] - confidence_interval[0])/2:.1f}%"
                )
                
                st.metric("Model Accuracy", f"{model_accuracy:.1f}%")
                st.metric("Confidence", f"{confidence_level}%")
    
    with col3:
        # Alert status
        if 'latest_prediction' in st.session_state:
            predicted_soh = st.session_state.latest_prediction['prediction']
            
            if enable_alerts:
                if predicted_soh < critical_threshold:
                    st.error("🚨 CRITICAL ALERT")
                    st.write("Immediate maintenance required")
                elif predicted_soh < warning_threshold:
                    st.warning("⚠️ WARNING")
                    st.write("Schedule maintenance soon")
                else:
                    st.success("✅ HEALTHY")
                    st.write("Normal operation")
    
    # Current telemetry display
    st.subheader("Current Telemetry Data")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("Current SoH", f"{latest_data['soh']:.1f}%")
    with col2:
        st.metric("Voltage", f"{latest_data['voltage']:.1f} V")
    with col3:
        st.metric("Current", f"{latest_data['current']:.1f} A")
    with col4:
        st.metric("Temperature", f"{latest_data['temperature']:.1f}°C")
    with col5:
        st.metric("Power", f"{latest_data['power']:.1f} kW")
    
    # Real-time updates
    if enable_realtime:
        # Create a placeholder for updates
        placeholder = st.empty()
        
        if st.button("▶️ Start Real-Time Monitoring"):
            for i in range(60):  # Run for 60 iterations
                with placeholder.container():
                    # Simulate new data (in real system, this would come from sensors)
                    simulated_data = latest_data.copy()
                    
                    # Add some realistic variations
                    simulated_data['soh'] += np.random.normal(0, 0.1)
                    simulated_data['voltage'] += np.random.normal(0, 2)
                    simulated_data['current'] += np.random.normal(0, 10)
                    simulated_data['temperature'] += np.random.normal(0, 1)
                    simulated_data['timestamp'] = datetime.now()
                    
                    # Update prediction
                    prediction_result = make_prediction_with_confidence(simulated_data, selected_model)
                    
                    if prediction_result:
                        st.write(f"**Live Update {i+1}** - {datetime.now().strftime('%H:%M:%S')}")
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Live SoH", f"{simulated_data['soh']:.1f}%")
                        with col2:
                            st.metric("Predicted SoH", f"{prediction_result['prediction']:.1f}%")
                        with col3:
                            st.metric("Confidence", f"±{(prediction_result['confidence_interval'][1] - prediction_result['confidence_interval'][0])/2:.1f}%")
                
                time.sleep(update_interval)

with tab2:
    st.header("Degradation Trend Analysis")
    
    if 'latest_prediction' in st.session_state:
        current_soh = st.session_state.latest_prediction['prediction']
        
        # Generate future trend
        dates, trend, upper_bound, lower_bound = generate_degradation_trend(
            current_soh, prediction_horizon
        )
        
        # Create trend visualization
        fig = go.Figure()
        
        # Add prediction line
        fig.add_trace(go.Scatter(
            x=dates,
            y=trend,
            mode='lines',
            name='Predicted SoH',
            line=dict(color='blue', width=3)
        ))
        
        # Add confidence interval
        fig.add_trace(go.Scatter(
            x=dates + dates[::-1],
            y=upper_bound + lower_bound[::-1],
            fill='toself',
            fillcolor='rgba(0, 0, 255, 0.2)',
            line=dict(color='rgba(255,255,255,0)'),
            name=f'{confidence_level}% Confidence Interval'
        ))
        
        # Add threshold lines
        fig.add_hline(
            y=warning_threshold, 
            line_dash="dash", 
            line_color="orange",
            annotation_text="Warning Threshold"
        )
        
        fig.add_hline(
            y=critical_threshold, 
            line_dash="dash", 
            line_color="red",
            annotation_text="Critical Threshold"
        )
        
        fig.update_layout(
            title=f"SoH Degradation Forecast - {prediction_horizon} Days",
            xaxis_title="Date",
            yaxis_title="State of Health (%)",
            hovermode='x unified'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Trend analysis metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            end_soh = trend[-1]
            st.metric("Predicted SoH (End)", f"{end_soh:.1f}%")
        
        with col2:
            total_degradation = current_soh - end_soh
            st.metric("Total Degradation", f"{total_degradation:.1f}%")
        
        with col3:
            daily_rate = total_degradation / prediction_horizon
            st.metric("Daily Degradation", f"{daily_rate:.3f}%/day")
        
        with col4:
            # Estimate time to warning threshold
            if daily_rate > 0:
                days_to_warning = max(0, (current_soh - warning_threshold) / daily_rate)
                st.metric("Days to Warning", f"{days_to_warning:.0f}")
            else:
                st.metric("Days to Warning", ">365")
        
        # Maintenance recommendations
        st.subheader("Maintenance Recommendations")
        
        if end_soh < critical_threshold:
            st.error(f"""
            🚨 **URGENT MAINTENANCE REQUIRED**
            
            The battery is predicted to reach critical levels ({critical_threshold}%) within {prediction_horizon} days.
            
            **Recommended Actions:**
            - Schedule immediate inspection
            - Consider battery replacement
            - Reduce operation intensity
            """)
        
        elif end_soh < warning_threshold:
            st.warning(f"""
            ⚠️ **MAINTENANCE RECOMMENDED**
            
            The battery will approach warning levels ({warning_threshold}%) within {prediction_horizon} days.
            
            **Recommended Actions:**
            - Schedule maintenance check
            - Monitor closely
            - Prepare for possible replacement
            """)
        
        else:
            st.success(f"""
            ✅ **BATTERY HEALTHY**
            
            The battery is expected to maintain good health above {warning_threshold}% for the next {prediction_horizon} days.
            
            **Recommended Actions:**
            - Continue regular monitoring
            - Maintain current operating conditions
            """)
    
    else:
        st.info("Please make a prediction first in the 'Live Prediction' tab.")

with tab3:
    st.header("Vehicle Diagnostics & Health Metrics")
    
    # Historical data for selected vehicle
    vehicle_history = fleet_data[fleet_data['vehicle_id'] == selected_vehicle].copy()
    vehicle_history = vehicle_history.sort_values('timestamp')
    
    if len(vehicle_history) > 0:
        # Diagnostic metrics
        st.subheader("Health Diagnostics")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # SoH trend over time
            fig = px.line(
                vehicle_history, 
                x='timestamp', 
                y='soh',
                title="Historical SoH Trend"
            )
            fig.add_hline(
                y=warning_threshold, 
                line_dash="dash", 
                line_color="orange"
            )
            fig.add_hline(
                y=critical_threshold, 
                line_dash="dash", 
                line_color="red"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Temperature vs SoH correlation
            fig = px.scatter(
                vehicle_history.tail(1000),  # Last 1000 points
                x='temperature',
                y='soh',
                title="Temperature Impact on SoH",
                trendline="ols"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Key performance indicators
        st.subheader("Key Performance Indicators")
        
        # Calculate KPIs
        initial_soh = vehicle_history['soh'].iloc[0]
        current_soh = vehicle_history['soh'].iloc[-1]
        total_degradation = initial_soh - current_soh
        
        time_span = (vehicle_history['timestamp'].iloc[-1] - 
                    vehicle_history['timestamp'].iloc[0]).days
        
        if time_span > 0:
            degradation_rate = (total_degradation / time_span) * 365  # Per year
        else:
            degradation_rate = 0
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Initial SoH", f"{initial_soh:.1f}%")
        with col2:
            st.metric("Current SoH", f"{current_soh:.1f}%")
        with col3:
            st.metric("Total Degradation", f"{total_degradation:.1f}%")
        with col4:
            st.metric("Annual Degradation Rate", f"{degradation_rate:.2f}%/year")
        
        # Operating conditions analysis
        st.subheader("Operating Conditions Analysis")
        
        # Statistics table
        operating_stats = vehicle_history[['temperature', 'voltage', 'current', 'power']].describe()
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Operating Statistics**")
            st.dataframe(operating_stats.round(2), use_container_width=True)
        
        with col2:
            # Distribution plots
            fig = px.box(
                vehicle_history.melt(
                    id_vars=['timestamp'],
                    value_vars=['temperature', 'voltage', 'current', 'power']
                ),
                x='variable',
                y='value',
                title="Operating Parameter Distributions"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Health score calculation
        st.subheader("Comprehensive Health Score")
        
        # Calculate various health indicators
        temp_score = 100 - min(100, abs(vehicle_history['temperature'].mean() - 25) * 2)
        voltage_stability = 100 - (vehicle_history['voltage'].std() / vehicle_history['voltage'].mean() * 100)
        soh_score = current_soh
        
        # Cycle health (based on DoD)
        avg_dod = vehicle_history['depth_of_discharge'].mean()
        cycle_score = 100 - (avg_dod * 50)  # Penalty for high DoD
        
        # Overall health score (weighted average)
        overall_health = (soh_score * 0.4 + temp_score * 0.2 + 
                         voltage_stability * 0.2 + cycle_score * 0.2)
        
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric("SoH Score", f"{soh_score:.1f}")
        with col2:
            st.metric("Temperature Score", f"{temp_score:.1f}")
        with col3:
            st.metric("Voltage Stability", f"{voltage_stability:.1f}")
        with col4:
            st.metric("Cycle Health", f"{cycle_score:.1f}")
        with col5:
            delta_color = "normal" if overall_health > 80 else "inverse"
            st.metric("Overall Health", f"{overall_health:.1f}", delta_color=delta_color)
    
    else:
        st.warning("No historical data available for the selected vehicle.")

with tab4:
    st.header("Prediction History & Model Performance")
    
    # Simulated prediction history (in real system, this would be stored)
    if 'prediction_history' not in st.session_state:
        st.session_state.prediction_history = []
    
    # Add current prediction to history if available
    if 'latest_prediction' in st.session_state and 'prediction_timestamp' in st.session_state:
        current_prediction = {
            'timestamp': st.session_state.prediction_timestamp,
            'vehicle_id': selected_vehicle,
            'model': selected_model,
            'predicted_soh': st.session_state.latest_prediction['prediction'],
            'confidence_interval': st.session_state.latest_prediction['confidence_interval'],
            'actual_soh': latest_data['soh']  # Current actual value
        }
        
        # Check if this prediction is already in history
        if not any(p['timestamp'] == current_prediction['timestamp'] and 
                  p['vehicle_id'] == current_prediction['vehicle_id'] 
                  for p in st.session_state.prediction_history):
            st.session_state.prediction_history.append(current_prediction)
    
    if st.session_state.prediction_history:
        # Convert to DataFrame
        history_df = pd.DataFrame(st.session_state.prediction_history)
        
        # Filter for selected vehicle
        vehicle_history = history_df[history_df['vehicle_id'] == selected_vehicle]
        
        if not vehicle_history.empty:
            st.subheader(f"Prediction History - {selected_vehicle}")
            
            # Display history table
            display_history = vehicle_history.copy()
            display_history['prediction_error'] = abs(
                display_history['predicted_soh'] - display_history['actual_soh']
            )
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.dataframe(
                    display_history[['timestamp', 'model', 'predicted_soh', 
                                   'actual_soh', 'prediction_error']].round(2),
                    use_container_width=True
                )
            
            with col2:
                # Prediction accuracy metrics
                avg_error = display_history['prediction_error'].mean()
                max_error = display_history['prediction_error'].max()
                predictions_count = len(display_history)
                
                st.metric("Average Error", f"{avg_error:.2f}%")
                st.metric("Max Error", f"{max_error:.2f}%")
                st.metric("Total Predictions", predictions_count)
            
            # Prediction accuracy over time
            if len(vehicle_history) > 1:
                fig = go.Figure()
                
                fig.add_trace(go.Scatter(
                    x=vehicle_history['timestamp'],
                    y=vehicle_history['predicted_soh'],
                    mode='lines+markers',
                    name='Predicted SoH',
                    line=dict(color='blue')
                ))
                
                fig.add_trace(go.Scatter(
                    x=vehicle_history['timestamp'],
                    y=vehicle_history['actual_soh'],
                    mode='lines+markers',
                    name='Actual SoH',
                    line=dict(color='red')
                ))
                
                fig.update_layout(
                    title="Prediction vs Actual SoH Over Time",
                    xaxis_title="Time",
                    yaxis_title="State of Health (%)"
                )
                
                st.plotly_chart(fig, use_container_width=True)
        
        # Model performance comparison
        st.subheader("Model Performance Comparison")
        
        if len(history_df) > 0:
            model_performance = history_df.groupby('model').agg({
                'prediction_error': ['mean', 'std', 'count']
            }).round(2)
            
            model_performance.columns = ['Mean Error', 'Std Error', 'Predictions']
            model_performance['Accuracy'] = 100 - model_performance['Mean Error']
            
            st.dataframe(model_performance, use_container_width=True)
    
    else:
        st.info("No prediction history available. Make some predictions to see the history.")
    
    # Export predictions
    st.subheader("Export Prediction Data")
    
    if st.session_state.prediction_history:
        export_df = pd.DataFrame(st.session_state.prediction_history)
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("📥 Export Prediction History"):
                csv_data = export_df.to_csv(index=False)
                st.download_button(
                    label="Download CSV",
                    data=csv_data,
                    file_name=f"prediction_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
        
        with col2:
            if st.button("🗑️ Clear History"):
                st.session_state.prediction_history = []
                st.rerun()

# Footer
st.markdown("---")
st.markdown("""
**Real-Time Prediction System** - Advanced ML-powered SoH forecasting for electric vehicle batteries.
Monitor live battery health, predict degradation trends, and receive maintenance alerts.
""")

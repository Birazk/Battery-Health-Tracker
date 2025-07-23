import streamlit as st
import pandas as pd
import numpy as np
from src.data_generator import EVBatteryDataGenerator
from src.data_processor import BatteryDataProcessor
from src.database import DatabaseManager
from utils.visualization import create_data_overview_plots
import plotly.express as px
import plotly.graph_objects as go

# Page configuration
st.set_page_config(page_title="Data Ingestion", page_icon="📊", layout="wide")

# Initialize components
@st.cache_resource
def get_data_components():
    return EVBatteryDataGenerator(), BatteryDataProcessor()

data_generator, data_processor = get_data_components()

# Initialize database manager
if 'db_manager' not in st.session_state:
    try:
        st.session_state.db_manager = DatabaseManager()
    except Exception as e:
        st.error(f"Database connection failed: {str(e)}")
        st.session_state.db_manager = None

st.title("📊 Data Ingestion Pipeline")
st.markdown("### Electric Vehicle Battery Telemetry Data Management")

# Sidebar for data generation parameters
with st.sidebar:
    st.header("Data Generation Settings")
    
    # Fleet configuration
    st.subheader("Fleet Configuration")
    num_vehicles = st.slider("Number of Vehicles", 10, 100, 30)
    data_days = st.slider("Data Collection Days", 30, 365, 180)
    
    # Vehicle type distribution
    st.subheader("Vehicle Type Distribution")
    car_pct = st.slider("Cars (%)", 0, 100, 40)
    bus_pct = st.slider("Buses (%)", 0, 100, 30)
    truck_pct = st.slider("Trucks (%)", 0, 100, 20)
    motorcycle_pct = st.slider("Motorcycles (%)", 0, 100, 10)
    
    # Normalize percentages
    total_pct = car_pct + bus_pct + truck_pct + motorcycle_pct
    if total_pct != 100:
        st.warning(f"Total percentage: {total_pct}%. Adjusting proportionally.")
    
    # Data quality settings
    st.subheader("Data Quality Simulation")
    add_missing = st.checkbox("Add Missing Values", False)
    add_outliers = st.checkbox("Add Outliers", False)
    
    if add_missing:
        missing_rate = st.slider("Missing Data Rate (%)", 0.0, 10.0, 2.0) / 100
    else:
        missing_rate = 0.0
    
    if add_outliers:
        outlier_rate = st.slider("Outlier Rate (%)", 0.0, 5.0, 1.0) / 100
    else:
        outlier_rate = 0.0

# Main content tabs
tab1, tab2, tab3, tab4 = st.tabs(["Data Generation", "Data Quality", "Data Overview", "Data Export"])

with tab1:
    st.header("Generate Fleet Data")
    
    # Database status
    if st.session_state.db_manager:
        st.success("✅ Database Connected - Data will be automatically saved")
        
        # Option to load existing data
        col_load, col_gen = st.columns([1, 2])
        with col_load:
            if st.button("📂 Load Existing Data from Database"):
                with st.spinner("Loading data from database..."):
                    existing_data = st.session_state.db_manager.load_fleet_data()
                    if existing_data is not None and not existing_data.empty:
                        st.session_state.fleet_data = existing_data
                        st.session_state.data_generated = True
                        st.success(f"✅ Loaded {len(existing_data):,} records from database")
                    else:
                        st.info("No existing data found in database")
    else:
        st.warning("⚠️ Database not connected - Data will only be stored in session")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        if st.button("🚀 Generate Fleet Data", type="primary"):
            with st.spinner("Generating fleet telemetry data..."):
                # Generate base fleet data
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Update data generator with custom distribution
                if total_pct > 0:
                    data_generator.vehicle_types = {
                        'car': data_generator.vehicle_types['car'],
                        'bus': data_generator.vehicle_types['bus'],
                        'truck': data_generator.vehicle_types['truck'],
                        'motorcycle': data_generator.vehicle_types['motorcycle']
                    }
                
                status_text.text("Generating vehicle profiles...")
                progress_bar.progress(25)
                
                # Generate fleet data
                fleet_data = data_generator.generate_fleet_data(num_vehicles, data_days)
                progress_bar.progress(50)
                
                status_text.text("Adding data quality variations...")
                if add_missing or add_outliers:
                    fleet_data = data_generator.add_data_quality_issues(
                        fleet_data, missing_rate, outlier_rate
                    )
                progress_bar.progress(75)
                
                # Store in session state and database
                st.session_state.fleet_data = fleet_data
                st.session_state.data_generated = True
                
                # Save to database
                status_text.text("Saving to database...")
                if st.session_state.db_manager:
                    success = st.session_state.db_manager.save_fleet_data(fleet_data)
                    if success:
                        st.session_state.db_manager.save_vehicle_profiles(data_generator.vehicle_profiles)
                
                progress_bar.progress(100)
                status_text.text("Data generation complete!")
                
                st.success(f"✅ Generated {len(fleet_data):,} telemetry records for {num_vehicles} vehicles over {data_days} days")
    
    with col2:
        if 'data_generated' in st.session_state and st.session_state.data_generated:
            st.metric("Records Generated", f"{len(st.session_state.fleet_data):,}")
            st.metric("Vehicles", len(st.session_state.fleet_data['vehicle_id'].unique()))
            st.metric("Time Span", f"{data_days} days")
    
    # Show sample data
    if 'fleet_data' in st.session_state:
        st.subheader("Sample Data Preview")
        sample_data = st.session_state.fleet_data.head(1000)
        st.dataframe(sample_data.head(10), use_container_width=True)
        
        # Basic statistics
        st.subheader("Basic Statistics")
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Numerical Statistics**")
            numerical_cols = ['soh', 'voltage', 'current', 'power', 'temperature']
            available_cols = [col for col in numerical_cols if col in sample_data.columns]
            st.dataframe(sample_data[available_cols].describe())
        
        with col2:
            st.write("**Vehicle Type Distribution**")
            vehicle_counts = sample_data['vehicle_type'].value_counts()
            fig = px.pie(values=vehicle_counts.values, names=vehicle_counts.index)
            st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.header("Data Quality Assessment")
    
    if 'fleet_data' in st.session_state:
        fleet_data = st.session_state.fleet_data
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            if st.button("🔍 Run Data Quality Check"):
                with st.spinner("Analyzing data quality..."):
                    quality_report = data_processor.validate_data_quality(fleet_data)
                    st.session_state.quality_report = quality_report
        
        with col2:
            if st.button("🧹 Clean Data"):
                with st.spinner("Cleaning data..."):
                    clean_data = data_processor.clean_data(fleet_data)
                    st.session_state.clean_data = clean_data
                    st.success("Data cleaned successfully!")
        
        # Display quality report
        if 'quality_report' in st.session_state:
            quality_report = st.session_state.quality_report
            
            # Overall quality metrics
            st.subheader("Data Quality Overview")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Records", quality_report['total_records'])
            with col2:
                st.metric("Duplicate Records", quality_report['duplicates'])
            with col3:
                st.metric("Time Gaps", quality_report['time_gaps'])
            with col4:
                overall_quality = data_processor.get_data_quality_summary()
                if isinstance(overall_quality, dict):
                    st.metric("Overall Quality", overall_quality['overall_quality'])
            
            # Missing values analysis
            st.subheader("Missing Values Analysis")
            missing_data = []
            for col, info in quality_report['missing_values'].items():
                if info['count'] > 0:
                    missing_data.append({
                        'Column': col,
                        'Missing Count': info['count'],
                        'Missing %': info['percentage']
                    })
            
            if missing_data:
                missing_df = pd.DataFrame(missing_data)
                fig = px.bar(missing_df, x='Column', y='Missing %', 
                           title="Missing Values by Column")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No missing values detected.")
            
            # Outliers analysis
            st.subheader("Outliers Analysis")
            outlier_data = []
            for col, info in quality_report['outliers'].items():
                if info['count'] > 0:
                    outlier_data.append({
                        'Column': col,
                        'Outlier Count': info['count'],
                        'Outlier %': info['percentage']
                    })
            
            if outlier_data:
                outlier_df = pd.DataFrame(outlier_data)
                fig = px.bar(outlier_df, x='Column', y='Outlier %', 
                           title="Outliers by Column")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No significant outliers detected.")
        
        # Anomaly detection
        st.subheader("Anomaly Detection")
        
        vehicle_ids = fleet_data['vehicle_id'].unique()
        selected_vehicle = st.selectbox("Select Vehicle for Anomaly Analysis", vehicle_ids)
        
        if st.button("🚨 Detect Anomalies"):
            with st.spinner("Detecting anomalies..."):
                anomalies = data_processor.detect_anomalies(fleet_data, selected_vehicle)
                
                if not anomalies.empty:
                    st.warning(f"Found {len(anomalies)} anomalies for vehicle {selected_vehicle}")
                    
                    # Group by severity
                    severity_counts = anomalies['severity'].value_counts()
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("High Severity", severity_counts.get('High', 0))
                    with col2:
                        st.metric("Medium Severity", severity_counts.get('Medium', 0))
                    with col3:
                        st.metric("Low Severity", severity_counts.get('Low', 0))
                    
                    # Show anomalies table
                    st.dataframe(anomalies, use_container_width=True)
                else:
                    st.success("No anomalies detected for the selected vehicle.")
    
    else:
        st.info("Please generate fleet data first in the 'Data Generation' tab.")

with tab3:
    st.header("Data Overview & Visualization")
    
    if 'fleet_data' in st.session_state:
        fleet_data = st.session_state.fleet_data
        
        # Fleet summary metrics
        st.subheader("Fleet Summary")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Vehicles", len(fleet_data['vehicle_id'].unique()))
        with col2:
            st.metric("Total Records", len(fleet_data))
        with col3:
            avg_soh = fleet_data['soh'].mean()
            st.metric("Average SoH", f"{avg_soh:.1f}%")
        with col4:
            date_range = (fleet_data['timestamp'].max() - fleet_data['timestamp'].min()).days
            st.metric("Data Span", f"{date_range} days")
        
        # Visualizations
        col1, col2 = st.columns(2)
        
        with col1:
            # SoH distribution by vehicle type
            fig = px.box(fleet_data, x='vehicle_type', y='soh',
                        title="SoH Distribution by Vehicle Type")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Temperature vs SoH scatter
            sample_data = fleet_data.sample(n=min(5000, len(fleet_data)))
            fig = px.scatter(sample_data, x='temperature', y='soh', 
                           color='vehicle_type',
                           title="Temperature vs State of Health")
            st.plotly_chart(fig, use_container_width=True)
        
        # Time series plot
        st.subheader("Time Series Analysis")
        
        # Select vehicles for time series
        selected_vehicles = st.multiselect(
            "Select Vehicles for Time Series",
            fleet_data['vehicle_id'].unique()[:10],  # Limit to first 10 for performance
            default=fleet_data['vehicle_id'].unique()[:3]
        )
        
        if selected_vehicles:
            ts_data = fleet_data[fleet_data['vehicle_id'].isin(selected_vehicles)]
            
            fig = go.Figure()
            
            for vehicle in selected_vehicles:
                vehicle_data = ts_data[ts_data['vehicle_id'] == vehicle].sort_values('timestamp')
                fig.add_trace(go.Scatter(
                    x=vehicle_data['timestamp'],
                    y=vehicle_data['soh'],
                    mode='lines',
                    name=vehicle,
                    line=dict(width=2)
                ))
            
            fig.update_layout(
                title="State of Health Trends Over Time",
                xaxis_title="Time",
                yaxis_title="State of Health (%)",
                hovermode='x unified'
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Correlation matrix
        st.subheader("Feature Correlation Matrix")
        
        correlation_cols = ['soh', 'voltage', 'current', 'power', 'temperature', 
                          'internal_resistance', 'cycle_count']
        available_cols = [col for col in correlation_cols if col in fleet_data.columns]
        
        if len(available_cols) > 1:
            corr_matrix = fleet_data[available_cols].corr()
            fig = px.imshow(corr_matrix, text_auto=True, aspect="auto",
                          title="Feature Correlation Matrix")
            st.plotly_chart(fig, use_container_width=True)
    
    else:
        st.info("Please generate fleet data first in the 'Data Generation' tab.")

with tab4:
    st.header("Data Export & Management")
    
    if 'fleet_data' in st.session_state:
        fleet_data = st.session_state.fleet_data
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Export Options")
            
            # Data format selection
            export_format = st.selectbox("Select Export Format", ["CSV", "JSON", "Parquet"])
            
            # Data subset selection
            export_option = st.radio(
                "Export Data Subset",
                ["All Data", "Specific Vehicle Type", "Specific Time Range", "Custom Selection"]
            )
            
            # Apply filters based on selection
            export_data = fleet_data.copy()
            
            if export_option == "Specific Vehicle Type":
                selected_type = st.selectbox("Vehicle Type", fleet_data['vehicle_type'].unique())
                export_data = export_data[export_data['vehicle_type'] == selected_type]
            
            elif export_option == "Specific Time Range":
                date_range = st.date_input(
                    "Select Date Range",
                    value=[fleet_data['timestamp'].min().date(), 
                          fleet_data['timestamp'].max().date()],
                    min_value=fleet_data['timestamp'].min().date(),
                    max_value=fleet_data['timestamp'].max().date()
                )
                if len(date_range) == 2:
                    start_date, end_date = date_range
                    export_data = export_data[
                        (export_data['timestamp'].dt.date >= start_date) &
                        (export_data['timestamp'].dt.date <= end_date)
                    ]
            
            elif export_option == "Custom Selection":
                # Vehicle selection
                selected_vehicles = st.multiselect(
                    "Select Vehicles",
                    fleet_data['vehicle_id'].unique(),
                    default=fleet_data['vehicle_id'].unique()[:5]
                )
                if selected_vehicles:
                    export_data = export_data[export_data['vehicle_id'].isin(selected_vehicles)]
                
                # Column selection
                selected_columns = st.multiselect(
                    "Select Columns",
                    fleet_data.columns.tolist(),
                    default=['timestamp', 'vehicle_id', 'vehicle_type', 'soh', 'voltage', 'temperature']
                )
                if selected_columns:
                    export_data = export_data[selected_columns]
            
            # Export preview
            st.write(f"**Export Preview** ({len(export_data):,} records)")
            st.dataframe(export_data.head(), use_container_width=True)
            
            # Download button
            if export_format == "CSV":
                csv_data = export_data.to_csv(index=False)
                st.download_button(
                    label="📥 Download CSV",
                    data=csv_data,
                    file_name=f"ev_battery_data_{export_option.lower().replace(' ', '_')}.csv",
                    mime="text/csv"
                )
            
            elif export_format == "JSON":
                json_data = export_data.to_json(orient='records', date_format='iso')
                st.download_button(
                    label="📥 Download JSON",
                    data=json_data,
                    file_name=f"ev_battery_data_{export_option.lower().replace(' ', '_')}.json",
                    mime="application/json"
                )
        
        with col2:
            st.subheader("Data Summary")
            
            # Export data statistics
            st.write("**Export Data Statistics**")
            st.write(f"- Records: {len(export_data):,}")
            st.write(f"- Vehicles: {len(export_data['vehicle_id'].unique()) if 'vehicle_id' in export_data.columns else 'N/A'}")
            st.write(f"- Columns: {len(export_data.columns)}")
            
            if 'timestamp' in export_data.columns:
                date_range = (export_data['timestamp'].max() - export_data['timestamp'].min()).days
                st.write(f"- Date Range: {date_range} days")
            
            # Storage recommendations
            st.subheader("Storage Recommendations")
            
            estimated_size_mb = len(export_data) * len(export_data.columns) * 8 / (1024 * 1024)  # Rough estimate
            
            st.info(f"""
            **Estimated Export Size:** {estimated_size_mb:.1f} MB
            
            **Format Recommendations:**
            - **CSV**: Best for spreadsheet applications
            - **JSON**: Best for web applications
            - **Parquet**: Best for big data processing (most efficient)
            """)
            
            # Data preparation for ML
            st.subheader("ML Preparation")
            
            if st.button("🤖 Prepare for ML Training"):
                # Store prepared data for ML training
                st.session_state.ml_ready_data = export_data
                st.success("✅ Data prepared for ML training! Navigate to 'Model Training' tab.")
    
    else:
        st.info("Please generate fleet data first in the 'Data Generation' tab.")

# Footer
st.markdown("---")
st.markdown("""
**Data Ingestion Pipeline** - Generate, validate, and prepare EV battery telemetry data for machine learning.
Use the tabs above to generate data, assess quality, explore patterns, and export for further analysis.
""")

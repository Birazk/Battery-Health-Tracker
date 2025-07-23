import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from src.database import DatabaseManager
from datetime import datetime, timedelta

# Page configuration
st.set_page_config(page_title="Fleet Dashboard", page_icon="🚗", layout="wide")

st.title("🚗 Fleet Monitoring Dashboard")
st.markdown("### Comprehensive multi-vehicle battery health monitoring across EV types")

# Check for required data
if 'fleet_data' not in st.session_state:
    st.error("❌ No fleet data available. Please generate data in the 'Data Ingestion' tab first.")
    st.stop()

fleet_data = st.session_state.fleet_data

# Initialize database manager
if 'db_manager' not in st.session_state:
    try:
        st.session_state.db_manager = DatabaseManager()
    except Exception as e:
        st.error(f"Database connection failed: {str(e)}")
        st.session_state.db_manager = None

# Sidebar for fleet filtering and settings
with st.sidebar:
    st.header("Fleet Filters")
    
    # Vehicle type filter
    available_types = fleet_data['vehicle_type'].unique()
    selected_types = st.multiselect(
        "Vehicle Types",
        available_types,
        default=available_types
    )
    
    # SoH range filter
    st.subheader("SoH Range Filter")
    min_soh, max_soh = st.slider(
        "State of Health Range (%)",
        float(fleet_data['soh'].min()),
        float(fleet_data['soh'].max()),
        (float(fleet_data['soh'].min()), float(fleet_data['soh'].max()))
    )
    
    # Age filter
    st.subheader("Vehicle Age Filter")
    min_age, max_age = st.slider(
        "Age Range (days)",
        int(fleet_data['age_days'].min()),
        int(fleet_data['age_days'].max()),
        (int(fleet_data['age_days'].min()), int(fleet_data['age_days'].max()))
    )
    
    # Time range filter
    st.subheader("Time Range")
    date_range = st.date_input(
        "Select Date Range",
        value=[fleet_data['timestamp'].min().date(), 
               fleet_data['timestamp'].max().date()],
        min_value=fleet_data['timestamp'].min().date(),
        max_value=fleet_data['timestamp'].max().date()
    )
    
    # Dashboard settings
    st.subheader("Dashboard Settings")
    auto_refresh = st.checkbox("Auto Refresh", False)
    if auto_refresh:
        refresh_interval = st.slider("Refresh Interval (seconds)", 5, 60, 30)
    
    show_predictions = st.checkbox("Show ML Predictions", True)
    
    # Alert thresholds
    st.subheader("Alert Thresholds")
    warning_threshold = st.slider("Warning Threshold (%)", 70, 95, 85)
    critical_threshold = st.slider("Critical Threshold (%)", 70, 90, 80)

# Apply filters
filtered_data = fleet_data[
    (fleet_data['vehicle_type'].isin(selected_types)) &
    (fleet_data['soh'] >= min_soh) &
    (fleet_data['soh'] <= max_soh) &
    (fleet_data['age_days'] >= min_age) &
    (fleet_data['age_days'] <= max_age)
].copy()

# Apply date filter
if len(date_range) == 2:
    start_date, end_date = date_range
    filtered_data = filtered_data[
        (filtered_data['timestamp'].dt.date >= start_date) &
        (filtered_data['timestamp'].dt.date <= end_date)
    ]

# Main dashboard content
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Fleet Overview",
    "Health Monitoring", 
    "Performance Analytics",
    "Maintenance Alerts",
    "Comparative Analysis"
])

def calculate_fleet_kpis(data):
    """Calculate key performance indicators for the fleet"""
    latest_data = data.groupby('vehicle_id').last()
    
    kpis = {
        'total_vehicles': len(latest_data),
        'avg_soh': latest_data['soh'].mean(),
        'healthy_vehicles': len(latest_data[latest_data['soh'] > warning_threshold]),
        'warning_vehicles': len(latest_data[
            (latest_data['soh'] <= warning_threshold) & 
            (latest_data['soh'] > critical_threshold)
        ]),
        'critical_vehicles': len(latest_data[latest_data['soh'] <= critical_threshold]),
        'avg_age': latest_data['age_days'].mean(),
        'total_capacity': latest_data['battery_capacity'].sum(),
        'avg_temperature': data['temperature'].mean()
    }
    
    return kpis, latest_data

with tab1:
    st.header("Fleet Overview")
    
    # Calculate KPIs
    kpis, latest_fleet_data = calculate_fleet_kpis(filtered_data)
    
    # Main KPI metrics
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("Total Vehicles", kpis['total_vehicles'])
    with col2:
        st.metric("Average SoH", f"{kpis['avg_soh']:.1f}%")
    with col3:
        healthy_pct = (kpis['healthy_vehicles'] / kpis['total_vehicles']) * 100 if kpis['total_vehicles'] > 0 else 0
        st.metric("Healthy Vehicles", f"{kpis['healthy_vehicles']} ({healthy_pct:.0f}%)")
    with col4:
        st.metric("Total Capacity", f"{kpis['total_capacity']:.0f} kWh")
    with col5:
        st.metric("Average Age", f"{kpis['avg_age']:.0f} days")
    
    # Alert status
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.success(f"✅ Healthy: {kpis['healthy_vehicles']}")
    with col2:
        st.warning(f"⚠️ Warning: {kpis['warning_vehicles']}")
    with col3:
        st.error(f"🚨 Critical: {kpis['critical_vehicles']}")
    
    # Fleet composition and status visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        # Fleet composition by vehicle type
        type_counts = latest_fleet_data['vehicle_type'].value_counts()
        fig = px.pie(
            values=type_counts.values,
            names=type_counts.index,
            title="Fleet Composition by Vehicle Type"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Health status distribution
        health_status = []
        for _, vehicle in latest_fleet_data.iterrows():
            if vehicle['soh'] > warning_threshold:
                health_status.append('Healthy')
            elif vehicle['soh'] > critical_threshold:
                health_status.append('Warning')
            else:
                health_status.append('Critical')
        
        health_counts = pd.Series(health_status).value_counts()
        colors = {'Healthy': 'green', 'Warning': 'orange', 'Critical': 'red'}
        
        fig = px.bar(
            x=health_counts.index,
            y=health_counts.values,
            title="Fleet Health Status Distribution",
            color=health_counts.index,
            color_discrete_map=colors
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Geographic/operational distribution (simulated)
    st.subheader("Fleet Distribution Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # SoH vs Age scatter plot
        fig = px.scatter(
            latest_fleet_data,
            x='age_days',
            y='soh',
            color='vehicle_type',
            size='battery_capacity',
            title="Battery Health vs Vehicle Age",
            labels={'age_days': 'Age (days)', 'soh': 'State of Health (%)'}
        )
        fig.add_hline(y=warning_threshold, line_dash="dash", line_color="orange")
        fig.add_hline(y=critical_threshold, line_dash="dash", line_color="red")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Capacity utilization by vehicle type
        capacity_by_type = latest_fleet_data.groupby('vehicle_type')['battery_capacity'].agg(['mean', 'sum', 'count'])
        
        fig = px.bar(
            x=capacity_by_type.index,
            y=capacity_by_type['sum'],
            title="Total Battery Capacity by Vehicle Type",
            labels={'x': 'Vehicle Type', 'y': 'Total Capacity (kWh)'}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Fleet summary table
    st.subheader("Vehicle Summary Table")
    
    # Prepare summary data
    summary_data = latest_fleet_data[['vehicle_id', 'vehicle_type', 'soh', 'age_days', 
                                     'battery_capacity', 'cycle_count', 'temperature']].copy()
    
    # Add health status
    summary_data['health_status'] = summary_data['soh'].apply(
        lambda x: 'Healthy' if x > warning_threshold else 'Warning' if x > critical_threshold else 'Critical'
    )
    
    # Sort by SoH (critical first)
    summary_data = summary_data.sort_values('soh')
    
    # Style the dataframe
    def color_health_status(val):
        if val == 'Critical':
            return 'background-color: #ffcccc'
        elif val == 'Warning':
            return 'background-color: #fff3cd'
        else:
            return 'background-color: #d4edda'
    
    styled_summary = summary_data.style.applymap(
        color_health_status, subset=['health_status']
    ).format({
        'soh': '{:.1f}%',
        'battery_capacity': '{:.1f} kWh',
        'cycle_count': '{:,.0f}',
        'temperature': '{:.1f}°C'
    })
    
    st.dataframe(styled_summary, use_container_width=True)

with tab2:
    st.header("Real-Time Health Monitoring")
    
    # Real-time metrics
    col1, col2, col3, col4 = st.columns(4)
    
    current_time = datetime.now()
    recent_data = filtered_data[
        filtered_data['timestamp'] >= (current_time - timedelta(hours=1))
    ]
    
    with col1:
        if len(recent_data) > 0:
            recent_avg_soh = recent_data['soh'].mean()
            st.metric("Current Avg SoH", f"{recent_avg_soh:.1f}%")
        else:
            st.metric("Current Avg SoH", "No recent data")
    
    with col2:
        if len(recent_data) > 0:
            recent_temp = recent_data['temperature'].mean()
            st.metric("Avg Temperature", f"{recent_temp:.1f}°C")
        else:
            st.metric("Avg Temperature", "No recent data")
    
    with col3:
        active_alerts = kpis['warning_vehicles'] + kpis['critical_vehicles']
        st.metric("Active Alerts", active_alerts)
    
    with col4:
        if len(recent_data) > 0:
            avg_power = recent_data['power'].mean()
            st.metric("Avg Power", f"{avg_power:.1f} kW")
        else:
            st.metric("Avg Power", "No recent data")
    
    # Time series monitoring
    st.subheader("Real-Time SoH Trends")
    
    # Select vehicles for monitoring
    vehicle_options = latest_fleet_data['vehicle_id'].tolist()
    selected_vehicles = st.multiselect(
        "Select Vehicles for Real-Time Monitoring",
        vehicle_options,
        default=vehicle_options[:5]  # Default to first 5
    )
    
    if selected_vehicles:
        # Create time series plot
        monitoring_data = filtered_data[
            filtered_data['vehicle_id'].isin(selected_vehicles)
        ].copy()
        
        fig = go.Figure()
        
        for vehicle_id in selected_vehicles:
            vehicle_data = monitoring_data[monitoring_data['vehicle_id'] == vehicle_id]
            vehicle_data = vehicle_data.sort_values('timestamp')
            
            fig.add_trace(go.Scatter(
                x=vehicle_data['timestamp'],
                y=vehicle_data['soh'],
                mode='lines+markers',
                name=vehicle_id,
                line=dict(width=2)
            ))
        
        # Add threshold lines
        fig.add_hline(
            y=warning_threshold,
            line_dash="dash",
            line_color="orange",
            annotation_text="Warning"
        )
        
        fig.add_hline(
            y=critical_threshold,
            line_dash="dash", 
            line_color="red",
            annotation_text="Critical"
        )
        
        fig.update_layout(
            title="Real-Time SoH Monitoring",
            xaxis_title="Time",
            yaxis_title="State of Health (%)",
            hovermode='x unified'
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Environmental conditions monitoring
    st.subheader("Environmental Conditions")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Temperature distribution
        fig = px.histogram(
            recent_data if len(recent_data) > 0 else filtered_data.tail(1000),
            x='temperature',
            nbins=30,
            title="Temperature Distribution (Recent Data)"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Temperature vs SoH correlation
        sample_data = filtered_data.sample(n=min(2000, len(filtered_data)))
        fig = px.scatter(
            sample_data,
            x='temperature',
            y='soh',
            color='vehicle_type',
            title="Temperature Impact on SoH",
            trendline="ols"
        )
        st.plotly_chart(fig, use_container_width=True)

with tab3:
    st.header("Performance Analytics")
    
    # Performance metrics by vehicle type
    st.subheader("Performance by Vehicle Type")
    
    type_performance = latest_fleet_data.groupby('vehicle_type').agg({
        'soh': ['mean', 'min', 'max', 'std'],
        'age_days': ['mean'],
        'cycle_count': ['mean'],
        'battery_capacity': ['mean'],
        'vehicle_id': ['count']
    }).round(2)
    
    # Flatten column names
    type_performance.columns = [f"{col[1]}_{col[0]}" if col[1] else col[0] for col in type_performance.columns]
    type_performance = type_performance.rename(columns={
        'mean_soh': 'Avg_SoH',
        'min_soh': 'Min_SoH', 
        'max_soh': 'Max_SoH',
        'std_soh': 'SoH_Std',
        'mean_age_days': 'Avg_Age',
        'mean_cycle_count': 'Avg_Cycles',
        'mean_battery_capacity': 'Avg_Capacity',
        'count_vehicle_id': 'Vehicle_Count'
    })
    
    st.dataframe(type_performance, use_container_width=True)
    
    # Performance visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        # SoH performance by vehicle type (box plot)
        fig = px.box(
            latest_fleet_data,
            x='vehicle_type',
            y='soh',
            title="SoH Distribution by Vehicle Type"
        )
        fig.add_hline(y=warning_threshold, line_dash="dash", line_color="orange")
        fig.add_hline(y=critical_threshold, line_dash="dash", line_color="red")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Degradation rate analysis
        degradation_data = []
        for vehicle_id in filtered_data['vehicle_id'].unique():
            vehicle_data = filtered_data[filtered_data['vehicle_id'] == vehicle_id].copy()
            vehicle_data = vehicle_data.sort_values('timestamp')
            
            if len(vehicle_data) > 1:
                initial_soh = vehicle_data['soh'].iloc[0]
                final_soh = vehicle_data['soh'].iloc[-1]
                days_span = (vehicle_data['timestamp'].iloc[-1] - 
                           vehicle_data['timestamp'].iloc[0]).days
                
                if days_span > 0:
                    degradation_rate = (initial_soh - final_soh) / days_span * 365  # Annualized
                    degradation_data.append({
                        'vehicle_id': vehicle_id,
                        'vehicle_type': vehicle_data['vehicle_type'].iloc[0],
                        'degradation_rate': degradation_rate
                    })
        
        if degradation_data:
            degradation_df = pd.DataFrame(degradation_data)
            
            fig = px.box(
                degradation_df,
                x='vehicle_type',
                y='degradation_rate',
                title="Annual Degradation Rate by Vehicle Type (%/year)"
            )
            st.plotly_chart(fig, use_container_width=True)
    
    # Efficiency analysis
    st.subheader("Operational Efficiency Analysis")
    
    # Calculate efficiency metrics
    latest_fleet_data['power_efficiency'] = np.where(
        latest_fleet_data['battery_capacity'] > 0,
        abs(latest_fleet_data['power']) / latest_fleet_data['battery_capacity'],
        0
    )
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Power efficiency by vehicle type
        fig = px.violin(
            latest_fleet_data,
            x='vehicle_type',
            y='power_efficiency',
            title="Power Efficiency Distribution by Vehicle Type"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Cycle count vs SoH
        fig = px.scatter(
            latest_fleet_data,
            x='cycle_count',
            y='soh',
            color='vehicle_type',
            title="Cycle Count vs State of Health",
            trendline="ols"
        )
        st.plotly_chart(fig, use_container_width=True)

with tab4:
    st.header("Maintenance Alerts & Recommendations")
    
    # Generate alerts
    critical_vehicles = latest_fleet_data[latest_fleet_data['soh'] <= critical_threshold]
    warning_vehicles = latest_fleet_data[
        (latest_fleet_data['soh'] <= warning_threshold) & 
        (latest_fleet_data['soh'] > critical_threshold)
    ]
    
    # Alert summary
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.error(f"🚨 Critical Alerts: {len(critical_vehicles)}")
        if len(critical_vehicles) > 0:
            st.write("Immediate action required")
    
    with col2:
        st.warning(f"⚠️ Warning Alerts: {len(warning_vehicles)}")
        if len(warning_vehicles) > 0:
            st.write("Schedule maintenance soon")
    
    with col3:
        healthy_count = len(latest_fleet_data) - len(critical_vehicles) - len(warning_vehicles)
        st.success(f"✅ Healthy Vehicles: {healthy_count}")
    
    # Critical alerts detail
    if len(critical_vehicles) > 0:
        st.subheader("🚨 Critical Maintenance Alerts")
        
        critical_display = critical_vehicles[['vehicle_id', 'vehicle_type', 'soh', 
                                           'age_days', 'cycle_count', 'temperature']].copy()
        critical_display['action_required'] = 'IMMEDIATE MAINTENANCE'
        critical_display['priority'] = 'HIGH'
        
        st.dataframe(critical_display.style.applymap(
            lambda x: 'background-color: #ffcccc', subset=['action_required']
        ), use_container_width=True)
    
    # Warning alerts detail
    if len(warning_vehicles) > 0:
        st.subheader("⚠️ Warning Maintenance Alerts")
        
        warning_display = warning_vehicles[['vehicle_id', 'vehicle_type', 'soh', 
                                         'age_days', 'cycle_count', 'temperature']].copy()
        warning_display['action_required'] = 'SCHEDULE MAINTENANCE'
        warning_display['priority'] = 'MEDIUM'
        
        st.dataframe(warning_display.style.applymap(
            lambda x: 'background-color: #fff3cd', subset=['action_required']
        ), use_container_width=True)
    
    # Maintenance recommendations
    st.subheader("Maintenance Recommendations")
    
    # Predictive maintenance based on patterns
    maintenance_recommendations = []
    
    for _, vehicle in latest_fleet_data.iterrows():
        recommendations = []
        
        # SoH-based recommendations
        if vehicle['soh'] <= critical_threshold:
            recommendations.append("Battery replacement required")
        elif vehicle['soh'] <= warning_threshold:
            recommendations.append("Battery inspection recommended")
        
        # Temperature-based recommendations
        if vehicle['temperature'] > 40:
            recommendations.append("Cooling system check")
        elif vehicle['temperature'] < 0:
            recommendations.append("Heating system check")
        
        # Cycle-based recommendations
        if vehicle['cycle_count'] > 5000:
            recommendations.append("High-cycle maintenance check")
        
        # Age-based recommendations
        if vehicle['age_days'] > 1000:  # ~3 years
            recommendations.append("Comprehensive inspection due to age")
        
        if recommendations:
            maintenance_recommendations.append({
                'vehicle_id': vehicle['vehicle_id'],
                'vehicle_type': vehicle['vehicle_type'],
                'soh': vehicle['soh'],
                'recommendations': '; '.join(recommendations),
                'urgency': 'High' if vehicle['soh'] <= critical_threshold else 
                          'Medium' if vehicle['soh'] <= warning_threshold else 'Low'
            })
    
    if maintenance_recommendations:
        maintenance_df = pd.DataFrame(maintenance_recommendations)
        maintenance_df = maintenance_df.sort_values('soh')  # Most critical first
        
        # Color code by urgency
        def color_urgency(val):
            if val == 'High':
                return 'background-color: #ffcccc'
            elif val == 'Medium':
                return 'background-color: #fff3cd'
            else:
                return 'background-color: #d4edda'
        
        styled_maintenance = maintenance_df.style.applymap(
            color_urgency, subset=['urgency']
        ).format({'soh': '{:.1f}%'})
        
        st.dataframe(styled_maintenance, use_container_width=True)
    
    # Maintenance schedule optimization
    st.subheader("Maintenance Schedule Optimization")
    
    if maintenance_recommendations:
        # Group by urgency and vehicle type
        urgency_counts = maintenance_df['urgency'].value_counts()
        type_urgency = maintenance_df.groupby(['vehicle_type', 'urgency']).size().unstack(fill_value=0)
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.pie(
                values=urgency_counts.values,
                names=urgency_counts.index,
                title="Maintenance Urgency Distribution",
                color_discrete_map={'High': 'red', 'Medium': 'orange', 'Low': 'green'}
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.bar(
                type_urgency,
                title="Maintenance Needs by Vehicle Type",
                color_discrete_map={'High': 'red', 'Medium': 'orange', 'Low': 'green'}
            )
            st.plotly_chart(fig, use_container_width=True)
    
    # Export maintenance data
    if st.button("📥 Export Maintenance Report"):
        if maintenance_recommendations:
            report_df = pd.DataFrame(maintenance_recommendations)
            csv_data = report_df.to_csv(index=False)
            
            st.download_button(
                label="Download Maintenance Report",
                data=csv_data,
                file_name=f"maintenance_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
        else:
            st.info("No maintenance recommendations to export.")

with tab5:
    st.header("Comparative Analysis")
    
    # Performance comparison across vehicle types
    st.subheader("Vehicle Type Performance Comparison")
    
    # Create comparison metrics
    comparison_metrics = latest_fleet_data.groupby('vehicle_type').agg({
        'soh': ['mean', 'std'],
        'age_days': 'mean',
        'cycle_count': 'mean', 
        'battery_capacity': 'mean',
        'temperature': 'mean',
        'vehicle_id': 'count'
    }).round(2)
    
    comparison_metrics.columns = ['_'.join(col).strip() for col in comparison_metrics.columns]
    
    # Ranking system
    comparison_metrics['health_score'] = comparison_metrics['soh_mean']
    comparison_metrics['reliability_score'] = 100 - comparison_metrics['soh_std']  # Lower std = higher reliability
    comparison_metrics['overall_score'] = (comparison_metrics['health_score'] + 
                                         comparison_metrics['reliability_score']) / 2
    
    # Sort by overall score
    comparison_metrics = comparison_metrics.sort_values('overall_score', ascending=False)
    
    st.dataframe(comparison_metrics, use_container_width=True)
    
    # Comparative visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        # Radar chart for vehicle type comparison
        categories = ['Health Score', 'Reliability Score', 'Overall Score']
        
        fig = go.Figure()
        
        for vehicle_type in comparison_metrics.index:
            values = [
                comparison_metrics.loc[vehicle_type, 'health_score'],
                comparison_metrics.loc[vehicle_type, 'reliability_score'],
                comparison_metrics.loc[vehicle_type, 'overall_score']
            ]
            
            fig.add_trace(go.Scatterpolar(
                r=values,
                theta=categories,
                fill='toself',
                name=vehicle_type
            ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 100]
                )
            ),
            title="Performance Comparison Radar Chart"
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Age vs Performance analysis
        fig = px.scatter(
            latest_fleet_data,
            x='age_days',
            y='soh',
            color='vehicle_type',
            size='battery_capacity',
            title="Age vs Performance by Vehicle Type",
            trendline="ols"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Best and worst performers
    st.subheader("Fleet Performance Highlights")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Best performing vehicle
        best_vehicle = latest_fleet_data.loc[latest_fleet_data['soh'].idxmax()]
        st.success(f"""
        **🏆 Best Performer**
        
        Vehicle: {best_vehicle['vehicle_id']}
        Type: {best_vehicle['vehicle_type']}
        SoH: {best_vehicle['soh']:.1f}%
        Age: {best_vehicle['age_days']} days
        """)
    
    with col2:
        # Most concerning vehicle
        worst_vehicle = latest_fleet_data.loc[latest_fleet_data['soh'].idxmin()]
        st.error(f"""
        **🚨 Most Concerning**
        
        Vehicle: {worst_vehicle['vehicle_id']}
        Type: {worst_vehicle['vehicle_type']}
        SoH: {worst_vehicle['soh']:.1f}%
        Age: {worst_vehicle['age_days']} days
        """)
    
    with col3:
        # Fleet average
        fleet_avg_soh = latest_fleet_data['soh'].mean()
        fleet_avg_age = latest_fleet_data['age_days'].mean()
        st.info(f"""
        **📊 Fleet Average**
        
        Average SoH: {fleet_avg_soh:.1f}%
        Average Age: {fleet_avg_age:.0f} days
        Total Vehicles: {len(latest_fleet_data)}
        Health Rate: {(kpis['healthy_vehicles']/kpis['total_vehicles']*100):.0f}%
        """)
    
    # Benchmarking against targets
    st.subheader("Performance Benchmarking")
    
    # Define benchmarks
    benchmarks = {
        'Target SoH': 90,
        'Warning Threshold': warning_threshold,
        'Critical Threshold': critical_threshold,
        'Fleet Average': fleet_avg_soh
    }
    
    # Performance against benchmarks by vehicle type
    benchmark_data = []
    for vehicle_type in latest_fleet_data['vehicle_type'].unique():
        type_data = latest_fleet_data[latest_fleet_data['vehicle_type'] == vehicle_type]
        avg_soh = type_data['soh'].mean()
        
        for benchmark_name, benchmark_value in benchmarks.items():
            benchmark_data.append({
                'Vehicle Type': vehicle_type,
                'Benchmark': benchmark_name,
                'Performance': avg_soh,
                'Target': benchmark_value,
                'Gap': avg_soh - benchmark_value
            })
    
    benchmark_df = pd.DataFrame(benchmark_data)
    
    # Heatmap of performance gaps
    pivot_benchmark = benchmark_df.pivot(index='Vehicle Type', 
                                       columns='Benchmark', 
                                       values='Gap')
    
    fig = px.imshow(
        pivot_benchmark.values,
        x=pivot_benchmark.columns,
        y=pivot_benchmark.index,
        color_continuous_scale='RdBu',
        title="Performance Gap Analysis (Positive = Above Target)"
    )
    
    st.plotly_chart(fig, use_container_width=True)

# Auto-refresh functionality
if auto_refresh:
    time.sleep(refresh_interval)
    st.rerun()

# Footer
st.markdown("---")
st.markdown(f"""
**Fleet Dashboard** - Comprehensive monitoring for {kpis['total_vehicles']} vehicles across {len(selected_types)} vehicle types.
Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
""")

import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def create_data_overview_plots(data):
    """
    Create comprehensive data overview plots for fleet data
    
    Args:
        data (pd.DataFrame): Fleet battery data
        
    Returns:
        dict: Dictionary containing various plotly figures
    """
    plots = {}
    
    # 1. SoH Distribution by Vehicle Type
    plots['soh_distribution'] = px.box(
        data, 
        x='vehicle_type', 
        y='soh',
        title="State of Health Distribution by Vehicle Type",
        color='vehicle_type'
    )
    
    # 2. Time Series of Average SoH
    daily_avg = data.groupby([data['timestamp'].dt.date, 'vehicle_type'])['soh'].mean().reset_index()
    daily_avg.columns = ['date', 'vehicle_type', 'avg_soh']
    
    plots['soh_timeseries'] = px.line(
        daily_avg,
        x='date',
        y='avg_soh',
        color='vehicle_type',
        title="Average State of Health Over Time"
    )
    
    # 3. Temperature vs SoH Correlation
    sample_data = data.sample(n=min(5000, len(data)))  # Sample for performance
    plots['temp_soh_correlation'] = px.scatter(
        sample_data,
        x='temperature',
        y='soh',
        color='vehicle_type',
        title="Temperature vs State of Health",
        trendline="ols"
    )
    
    # 4. Battery Capacity Distribution
    plots['capacity_distribution'] = px.histogram(
        data,
        x='battery_capacity',
        color='vehicle_type',
        title="Battery Capacity Distribution by Vehicle Type",
        nbins=30
    )
    
    # 5. Cycle Count vs SoH
    plots['cycles_soh'] = px.scatter(
        sample_data,
        x='cycle_count',
        y='soh',
        color='vehicle_type',
        size='battery_capacity',
        title="Cycle Count vs State of Health",
        trendline="ols"
    )
    
    return plots

def create_fleet_status_dashboard(data, warning_threshold=85, critical_threshold=80):
    """
    Create fleet status dashboard visualizations
    
    Args:
        data (pd.DataFrame): Fleet data
        warning_threshold (float): Warning SoH threshold
        critical_threshold (float): Critical SoH threshold
        
    Returns:
        dict: Dashboard visualizations
    """
    dashboard = {}
    
    # Get latest data for each vehicle
    latest_data = data.groupby('vehicle_id').last().reset_index()
    
    # 1. Health Status Distribution
    def get_health_status(soh):
        if soh <= critical_threshold:
            return 'Critical'
        elif soh <= warning_threshold:
            return 'Warning'
        else:
            return 'Healthy'
    
    latest_data['health_status'] = latest_data['soh'].apply(get_health_status)
    status_counts = latest_data['health_status'].value_counts()
    
    dashboard['health_status'] = px.pie(
        values=status_counts.values,
        names=status_counts.index,
        title="Fleet Health Status Distribution",
        color_discrete_map={
            'Healthy': 'green',
            'Warning': 'orange', 
            'Critical': 'red'
        }
    )
    
    # 2. SoH Heatmap by Vehicle Type and Age
    latest_data['age_group'] = pd.cut(
        latest_data['age_days'], 
        bins=5, 
        labels=['Very New', 'New', 'Medium', 'Old', 'Very Old']
    )
    
    heatmap_data = latest_data.groupby(['vehicle_type', 'age_group'])['soh'].mean().unstack()
    
    dashboard['soh_heatmap'] = px.imshow(
        heatmap_data.values,
        x=heatmap_data.columns,
        y=heatmap_data.index,
        color_continuous_scale='RdYlGn',
        title="Average SoH by Vehicle Type and Age Group"
    )
    
    # 3. Geographic/Operational Distribution (simulated)
    # In real implementation, this would use actual geographic data
    latest_data['region'] = np.random.choice(
        ['North', 'South', 'East', 'West', 'Central'], 
        size=len(latest_data)
    )
    
    region_stats = latest_data.groupby('region').agg({
        'soh': 'mean',
        'vehicle_id': 'count',
        'temperature': 'mean'
    }).round(2)
    
    dashboard['regional_performance'] = px.bar(
        x=region_stats.index,
        y=region_stats['soh'],
        title="Average SoH by Region",
        color=region_stats['soh'],
        color_continuous_scale='RdYlGn'
    )
    
    return dashboard

def create_prediction_visualization(actual, predicted, model_name, confidence_intervals=None):
    """
    Create prediction vs actual visualization
    
    Args:
        actual (array-like): Actual values
        predicted (array-like): Predicted values  
        model_name (str): Name of the model
        confidence_intervals (tuple): Lower and upper confidence bounds
        
    Returns:
        plotly.graph_objects.Figure: Prediction visualization
    """
    fig = go.Figure()
    
    # Scatter plot of predictions vs actual
    fig.add_trace(go.Scatter(
        x=actual,
        y=predicted,
        mode='markers',
        name='Predictions',
        marker=dict(
            size=6,
            opacity=0.6,
            color='blue'
        )
    ))
    
    # Perfect prediction line
    min_val = min(min(actual), min(predicted))
    max_val = max(max(actual), max(predicted))
    
    fig.add_trace(go.Scatter(
        x=[min_val, max_val],
        y=[min_val, max_val],
        mode='lines',
        name='Perfect Prediction',
        line=dict(color='red', dash='dash', width=2)
    ))
    
    # Confidence intervals if provided
    if confidence_intervals:
        lower_bound, upper_bound = confidence_intervals
        fig.add_trace(go.Scatter(
            x=actual,
            y=lower_bound,
            mode='lines',
            name='Lower Bound',
            line=dict(color='gray', dash='dot'),
            showlegend=False
        ))
        fig.add_trace(go.Scatter(
            x=actual,
            y=upper_bound,
            mode='lines',
            name='Upper Bound',
            line=dict(color='gray', dash='dot'),
            fill='tonexty',
            fillcolor='rgba(128,128,128,0.2)'
        ))
    
    fig.update_layout(
        title=f"{model_name}: Predicted vs Actual SoH",
        xaxis_title="Actual SoH (%)",
        yaxis_title="Predicted SoH (%)",
        template="plotly_white",
        width=600,
        height=500
    )
    
    return fig

def create_feature_importance_plot(importance_df, model_name, top_n=15):
    """
    Create feature importance visualization
    
    Args:
        importance_df (pd.DataFrame): DataFrame with 'feature' and 'importance' columns
        model_name (str): Name of the model
        top_n (int): Number of top features to display
        
    Returns:
        plotly.graph_objects.Figure: Feature importance plot
    """
    # Get top N features
    top_features = importance_df.head(top_n)
    
    fig = px.bar(
        top_features,
        x='importance',
        y='feature',
        orientation='h',
        title=f"Top {top_n} Feature Importance - {model_name}",
        color='importance',
        color_continuous_scale='viridis'
    )
    
    fig.update_layout(
        yaxis={'categoryorder': 'total ascending'},
        template="plotly_white",
        height=max(400, top_n * 25)  # Dynamic height based on number of features
    )
    
    return fig

def create_degradation_trend_plot(dates, soh_values, predicted_trend=None, 
                                 confidence_bounds=None, thresholds=None):
    """
    Create battery degradation trend visualization
    
    Args:
        dates (list): List of dates
        soh_values (list): Historical SoH values
        predicted_trend (list): Future predicted SoH values
        confidence_bounds (tuple): (lower_bound, upper_bound) for predictions
        thresholds (dict): Dictionary with 'warning' and 'critical' thresholds
        
    Returns:
        plotly.graph_objects.Figure: Degradation trend plot
    """
    fig = go.Figure()
    
    # Historical data
    fig.add_trace(go.Scatter(
        x=dates[:len(soh_values)],
        y=soh_values,
        mode='lines+markers',
        name='Historical SoH',
        line=dict(color='blue', width=2),
        marker=dict(size=4)
    ))
    
    # Predicted trend
    if predicted_trend:
        pred_dates = dates[len(soh_values)-1:]  # Start from last historical point
        fig.add_trace(go.Scatter(
            x=pred_dates,
            y=[soh_values[-1]] + predicted_trend,  # Connect to last historical point
            mode='lines',
            name='Predicted Trend',
            line=dict(color='orange', width=2, dash='dash')
        ))
        
        # Confidence bounds
        if confidence_bounds:
            lower_bound, upper_bound = confidence_bounds
            fig.add_trace(go.Scatter(
                x=pred_dates + pred_dates[::-1],
                y=[soh_values[-1]] + upper_bound + ([soh_values[-1]] + lower_bound)[::-1],
                fill='toself',
                fillcolor='rgba(255,165,0,0.2)',
                line=dict(color='rgba(255,255,255,0)'),
                name='Confidence Interval'
            ))
    
    # Threshold lines
    if thresholds:
        if 'warning' in thresholds:
            fig.add_hline(
                y=thresholds['warning'],
                line_dash="dash",
                line_color="orange",
                annotation_text="Warning Threshold"
            )
        
        if 'critical' in thresholds:
            fig.add_hline(
                y=thresholds['critical'],
                line_dash="dash",
                line_color="red", 
                annotation_text="Critical Threshold"
            )
    
    fig.update_layout(
        title="Battery State of Health Degradation Trend",
        xaxis_title="Date",
        yaxis_title="State of Health (%)",
        template="plotly_white",
        hovermode='x unified'
    )
    
    return fig

def create_multi_vehicle_comparison(vehicle_data_dict, metric='soh'):
    """
    Create comparison plot for multiple vehicles
    
    Args:
        vehicle_data_dict (dict): Dictionary with vehicle_id as key and data as value
        metric (str): Metric to compare ('soh', 'temperature', etc.)
        
    Returns:
        plotly.graph_objects.Figure: Multi-vehicle comparison plot
    """
    fig = go.Figure()
    
    colors = px.colors.qualitative.Set1
    
    for i, (vehicle_id, data) in enumerate(vehicle_data_dict.items()):
        color = colors[i % len(colors)]
        
        fig.add_trace(go.Scatter(
            x=data['timestamp'],
            y=data[metric],
            mode='lines',
            name=vehicle_id,
            line=dict(color=color, width=2)
        ))
    
    fig.update_layout(
        title=f"{metric.upper()} Comparison Across Vehicles",
        xaxis_title="Time",
        yaxis_title=f"{metric.upper()} (%)" if metric == 'soh' else metric.title(),
        template="plotly_white",
        hovermode='x unified'
    )
    
    return fig

def create_correlation_heatmap(data, features=None):
    """
    Create correlation heatmap for battery features
    
    Args:
        data (pd.DataFrame): Battery data
        features (list): List of features to include in correlation
        
    Returns:
        plotly.graph_objects.Figure: Correlation heatmap
    """
    if features is None:
        features = ['soh', 'voltage', 'current', 'power', 'temperature', 
                   'internal_resistance', 'cycle_count', 'age_days']
    
    # Select available features
    available_features = [f for f in features if f in data.columns]
    
    # Calculate correlation matrix
    corr_matrix = data[available_features].corr()
    
    fig = px.imshow(
        corr_matrix.values,
        x=corr_matrix.columns,
        y=corr_matrix.index,
        color_continuous_scale='RdBu',
        aspect="auto",
        title="Feature Correlation Matrix"
    )
    
    # Add correlation values as text
    fig.update_traces(
        text=corr_matrix.values.round(2),
        texttemplate="%{text}",
        textfont={"size": 10}
    )
    
    return fig

def create_real_time_gauge(current_value, min_val=70, max_val=100, 
                          warning_threshold=85, critical_threshold=80, title="State of Health"):
    """
    Create a real-time gauge chart for SoH monitoring
    
    Args:
        current_value (float): Current SoH value
        min_val (float): Minimum gauge value
        max_val (float): Maximum gauge value  
        warning_threshold (float): Warning threshold
        critical_threshold (float): Critical threshold
        title (str): Gauge title
        
    Returns:
        plotly.graph_objects.Figure: Gauge chart
    """
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=current_value,
        title={'text': title},
        delta={'reference': warning_threshold},
        gauge={
            'axis': {'range': [min_val, max_val]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [min_val, critical_threshold], 'color': "lightcoral"},
                {'range': [critical_threshold, warning_threshold], 'color': "lightyellow"},
                {'range': [warning_threshold, max_val], 'color': "lightgreen"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': warning_threshold
            }
        }
    ))
    
    fig.update_layout(height=300)
    
    return fig

def create_maintenance_timeline(maintenance_schedule):
    """
    Create maintenance timeline visualization
    
    Args:
        maintenance_schedule (pd.DataFrame): DataFrame with maintenance data
        
    Returns:
        plotly.graph_objects.Figure: Timeline visualization
    """
    fig = px.timeline(
        maintenance_schedule,
        x_start="start_date",
        x_end="end_date", 
        y="vehicle_id",
        color="urgency",
        title="Maintenance Schedule Timeline",
        color_discrete_map={
            'High': 'red',
            'Medium': 'orange',
            'Low': 'green'
        }
    )
    
    fig.update_layout(
        xaxis_title="Date",
        yaxis_title="Vehicle ID",
        template="plotly_white"
    )
    
    return fig

def create_performance_radar_chart(metrics_dict, model_name):
    """
    Create radar chart for model performance metrics
    
    Args:
        metrics_dict (dict): Dictionary of metrics and values
        model_name (str): Name of the model
        
    Returns:
        plotly.graph_objects.Figure: Radar chart
    """
    categories = list(metrics_dict.keys())
    values = list(metrics_dict.values())
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself',
        name=model_name,
        line_color='blue'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100]
            )
        ),
        title=f"Performance Radar Chart - {model_name}",
        template="plotly_white"
    )
    
    return fig

# Utility functions for data processing and visualization support

def prepare_time_series_data(data, vehicle_id=None, resample_freq='H'):
    """
    Prepare time series data for visualization
    
    Args:
        data (pd.DataFrame): Raw data
        vehicle_id (str): Specific vehicle ID or None for all
        resample_freq (str): Resampling frequency
        
    Returns:
        pd.DataFrame: Processed time series data
    """
    df = data.copy()
    
    if vehicle_id:
        df = df[df['vehicle_id'] == vehicle_id]
    
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.set_index('timestamp')
    
    # Resample if needed
    if resample_freq:
        df = df.resample(resample_freq).mean()
    
    return df.reset_index()

def calculate_fleet_statistics(data):
    """
    Calculate comprehensive fleet statistics
    
    Args:
        data (pd.DataFrame): Fleet data
        
    Returns:
        dict: Fleet statistics
    """
    latest_data = data.groupby('vehicle_id').last()
    
    stats = {
        'total_vehicles': len(latest_data),
        'avg_soh': latest_data['soh'].mean(),
        'min_soh': latest_data['soh'].min(),
        'max_soh': latest_data['soh'].max(),
        'std_soh': latest_data['soh'].std(),
        'avg_age_days': latest_data['age_days'].mean(),
        'total_capacity': latest_data['battery_capacity'].sum(),
        'avg_cycles': latest_data['cycle_count'].mean(),
        'vehicle_types': latest_data['vehicle_type'].value_counts().to_dict()
    }
    
    return stats

def export_visualization_data(fig, filename):
    """
    Export visualization data for external use
    
    Args:
        fig (plotly.graph_objects.Figure): Plotly figure
        filename (str): Export filename
        
    Returns:
        str: Success message
    """
    try:
        # Export as HTML
        fig.write_html(f"{filename}.html")
        
        # Export as PNG (requires kaleido)
        try:
            fig.write_image(f"{filename}.png")
        except:
            pass  # Skip if kaleido not available
        
        return f"Visualization exported as {filename}.html"
    
    except Exception as e:
        return f"Export failed: {str(e)}"

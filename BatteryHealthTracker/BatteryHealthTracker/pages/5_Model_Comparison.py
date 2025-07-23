import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from src.database import DatabaseManager
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import seaborn as sns
import matplotlib.pyplot as plt

# Page configuration
st.set_page_config(page_title="Model Comparison", page_icon="📈", layout="wide")

st.title("📈 Model Comparison & Performance Analysis")
st.markdown("### Comprehensive evaluation of ML algorithms for battery SoH prediction")

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

# Sidebar for comparison settings
with st.sidebar:
    st.header("Comparison Settings")
    
    # Model selection for comparison
    available_models = list(ml_models.model_scores.keys())
    selected_models = st.multiselect(
        "Select Models to Compare",
        available_models,
        default=available_models
    )
    
    # Metrics selection
    st.subheader("Evaluation Metrics")
    show_accuracy = st.checkbox("Accuracy Percentage", True)
    show_r2 = st.checkbox("R² Score", True)
    show_mae = st.checkbox("Mean Absolute Error", True)
    show_rmse = st.checkbox("Root Mean Square Error", True)
    
    # Comparison criteria
    st.subheader("Comparison Criteria")
    sort_by = st.selectbox(
        "Sort Models By",
        ["Accuracy (%)", "R² Score", "MAE (lower is better)", "RMSE (lower is better)"]
    )
    
    # Vehicle type analysis
    st.subheader("Vehicle Type Analysis")
    analyze_by_type = st.checkbox("Analyze Performance by Vehicle Type", True)
    
    # Statistical analysis
    st.subheader("Statistical Analysis")
    show_confidence_intervals = st.checkbox("Confidence Intervals", True)
    statistical_tests = st.checkbox("Statistical Significance Tests", False)

# Main content tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Overall Comparison",
    "Detailed Metrics", 
    "Prediction Analysis",
    "Feature Importance",
    "Model Selection Guide"
])

def get_model_predictions():
    """Get predictions from all trained models"""
    if hasattr(ml_models, 'test_data'):
        return ml_models.test_data
    return None

def calculate_additional_metrics(y_true, y_pred):
    """Calculate additional performance metrics"""
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    
    # Mean Absolute Percentage Error
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    
    # Accuracy as percentage (100% - normalized MAE)
    accuracy = max(0, (1 - mae/100) * 100)
    
    return {
        'MAE': mae,
        'MSE': mse,
        'RMSE': rmse,
        'R²': r2,
        'MAPE': mape,
        'Accuracy (%)': accuracy
    }

with tab1:
    st.header("Overall Model Performance Comparison")
    
    # Get model comparison data
    comparison_df = ml_models.get_model_comparison()
    
    if not comparison_df.empty:
        # Filter selected models
        if selected_models:
            comparison_df = comparison_df[comparison_df['Model'].isin(selected_models)]
        
        # Display comparison table
        st.subheader("Performance Summary Table")
        
        # Style the dataframe
        def highlight_best_worst(s):
            if s.name == 'Accuracy (%)':
                is_max = s == s.max()
                is_min = s == s.min()
                return ['background-color: lightgreen' if v else 
                       'background-color: lightcoral' if w else '' 
                       for v, w in zip(is_max, is_min)]
            elif s.name in ['MAE', 'RMSE']:
                is_min = s == s.min()  # Lower is better
                is_max = s == s.max()
                return ['background-color: lightgreen' if v else 
                       'background-color: lightcoral' if w else '' 
                       for v, w in zip(is_min, is_max)]
            elif s.name == 'Test R²':
                is_max = s == s.max()
                is_min = s == s.min()
                return ['background-color: lightgreen' if v else 
                       'background-color: lightcoral' if w else '' 
                       for v, w in zip(is_max, is_min)]
            return ['' for _ in s]
        
        styled_comparison = comparison_df.style.apply(highlight_best_worst, axis=0)
        st.dataframe(styled_comparison, use_container_width=True)
        
        # Performance visualization
        col1, col2 = st.columns(2)
        
        with col1:
            # Accuracy comparison bar chart
            fig = px.bar(
                comparison_df,
                x='Model',
                y='Accuracy (%)',
                title="Model Accuracy Comparison",
                color='Accuracy (%)',
                color_continuous_scale='RdYlGn'
            )
            fig.add_hline(y=95, line_dash="dash", line_color="red", 
                         annotation_text="95% Target")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Multi-metric radar chart
            if len(comparison_df) > 0:
                fig = go.Figure()
                
                # Normalize metrics for radar chart (0-100 scale)
                normalized_df = comparison_df.copy()
                normalized_df['Accuracy_norm'] = normalized_df['Accuracy (%)']
                normalized_df['R2_norm'] = normalized_df['Test R²'] * 100
                normalized_df['MAE_norm'] = 100 - (normalized_df['MAE'] / normalized_df['MAE'].max()) * 100
                normalized_df['RMSE_norm'] = 100 - (normalized_df['RMSE'] / normalized_df['RMSE'].max()) * 100
                
                categories = ['Accuracy', 'R² Score', 'MAE (inverted)', 'RMSE (inverted)']
                
                for _, model_row in normalized_df.iterrows():
                    values = [
                        model_row['Accuracy_norm'],
                        model_row['R2_norm'],
                        model_row['MAE_norm'],
                        model_row['RMSE_norm']
                    ]
                    
                    fig.add_trace(go.Scatterpolar(
                        r=values,
                        theta=categories,
                        fill='toself',
                        name=model_row['Model']
                    ))
                
                fig.update_layout(
                    polar=dict(
                        radialaxis=dict(
                            visible=True,
                            range=[0, 100]
                        )
                    ),
                    title="Multi-Metric Performance Comparison"
                )
                
                st.plotly_chart(fig, use_container_width=True)
        
        # Key insights
        st.subheader("Key Performance Insights")
        
        best_accuracy_model = comparison_df.loc[comparison_df['Accuracy (%)'].idxmax()]
        best_r2_model = comparison_df.loc[comparison_df['Test R²'].idxmax()]
        best_mae_model = comparison_df.loc[comparison_df['MAE'].idxmin()]
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.success(f"""
            **🎯 Highest Accuracy**
            
            Model: {best_accuracy_model['Model']}
            Accuracy: {best_accuracy_model['Accuracy (%)']:.1f}%
            Status: {'✅ Target Met' if best_accuracy_model['Accuracy (%)'] > 95 else '❌ Below Target'}
            """)
        
        with col2:
            st.info(f"""
            **📊 Best R² Score**
            
            Model: {best_r2_model['Model']}
            R² Score: {best_r2_model['Test R²']:.4f}
            Explains: {best_r2_model['Test R²']*100:.1f}% of variance
            """)
        
        with col3:
            st.success(f"""
            **🎯 Lowest Error**
            
            Model: {best_mae_model['Model']}
            MAE: {best_mae_model['MAE']:.2f}%
            Precision: ±{best_mae_model['MAE']:.1f}% typical error
            """)
        
        # Model ranking
        st.subheader("Model Ranking")
        
        # Create ranking based on selected criterion
        if sort_by == "Accuracy (%)":
            ranking_df = comparison_df.sort_values('Accuracy (%)', ascending=False)
        elif sort_by == "R² Score":
            ranking_df = comparison_df.sort_values('Test R²', ascending=False)
        elif sort_by == "MAE (lower is better)":
            ranking_df = comparison_df.sort_values('MAE', ascending=True)
        else:  # RMSE
            ranking_df = comparison_df.sort_values('RMSE', ascending=True)
        
        ranking_df['Rank'] = range(1, len(ranking_df) + 1)
        
        st.dataframe(
            ranking_df[['Rank', 'Model', 'Accuracy (%)', 'Test R²', 'MAE', 'RMSE', 'Target Met (>95%)']],
            use_container_width=True
        )

with tab2:
    st.header("Detailed Performance Metrics")
    
    # Get test data and predictions
    test_data = get_model_predictions()
    
    if test_data:
        y_test = test_data['y_test']
        predictions = test_data['predictions']
        
        # Calculate detailed metrics for each model
        detailed_metrics = []
        
        for model_name in selected_models:
            if model_name in predictions:
                y_pred = predictions[model_name]
                
                # Handle NaN values (especially for LSTM)
                if isinstance(y_pred, np.ndarray):
                    mask = ~np.isnan(y_pred)
                    if mask.sum() > 0:
                        y_test_clean = y_test[mask]
                        y_pred_clean = y_pred[mask]
                    else:
                        continue
                else:
                    y_test_clean = y_test
                    y_pred_clean = y_pred
                
                metrics = calculate_additional_metrics(y_test_clean, y_pred_clean)
                metrics['Model'] = model_name
                metrics['Sample Size'] = len(y_test_clean)
                detailed_metrics.append(metrics)
        
        if detailed_metrics:
            detailed_df = pd.DataFrame(detailed_metrics)
            
            # Display detailed metrics
            st.subheader("Comprehensive Metrics Table")
            st.dataframe(detailed_df.round(4), use_container_width=True)
            
            # Metrics visualization
            col1, col2 = st.columns(2)
            
            with col1:
                # Error metrics comparison
                error_metrics = detailed_df[['Model', 'MAE', 'RMSE', 'MAPE']].melt(
                    id_vars=['Model'], var_name='Metric', value_name='Value'
                )
                
                fig = px.bar(
                    error_metrics,
                    x='Model',
                    y='Value',
                    color='Metric',
                    barmode='group',
                    title="Error Metrics Comparison (Lower is Better)"
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Performance metrics comparison  
                perf_metrics = detailed_df[['Model', 'R²', 'Accuracy (%)']].copy()
                perf_metrics['R² (%)'] = perf_metrics['R²'] * 100
                
                perf_melted = perf_metrics[['Model', 'Accuracy (%)', 'R² (%)']].melt(
                    id_vars=['Model'], var_name='Metric', value_name='Value'
                )
                
                fig = px.bar(
                    perf_melted,
                    x='Model',
                    y='Value',
                    color='Metric',
                    barmode='group',
                    title="Performance Metrics Comparison (Higher is Better)"
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Statistical analysis
            if statistical_tests and len(detailed_metrics) > 1:
                st.subheader("Statistical Significance Analysis")
                
                # Perform pairwise comparisons (simplified)
                st.info("""
                **Statistical Analysis Summary:**
                
                For robust model comparison, consider:
                - Cross-validation results
                - Bootstrap confidence intervals  
                - McNemar's test for classification
                - Wilcoxon signed-rank test for predictions
                
                *Note: Full statistical testing requires multiple validation runs.*
                """)
    
    # Performance by vehicle type (if available)
    if analyze_by_type and 'engineered_data' in st.session_state:
        st.subheader("Performance by Vehicle Type")
        
        engineered_data = st.session_state.engineered_data
        
        if test_data and 'vehicle_type' in engineered_data.columns:
            # Match test indices with vehicle types
            X_test = test_data['X_test']
            test_indices = X_test.index
            test_vehicle_data = engineered_data.loc[test_indices]
            
            if 'vehicle_type' in test_vehicle_data.columns:
                vehicle_performance = []
                
                for vehicle_type in test_vehicle_data['vehicle_type'].unique():
                    type_mask = test_vehicle_data['vehicle_type'] == vehicle_type
                    type_indices = test_vehicle_data[type_mask].index
                    
                    y_true_type = y_test.loc[type_indices]
                    
                    for model_name in selected_models:
                        if model_name in predictions:
                            y_pred = predictions[model_name]
                            
                            if isinstance(y_pred, np.ndarray):
                                if len(y_pred) == len(y_test):
                                    y_pred_type = y_pred[type_mask.values]
                                else:
                                    continue
                            else:
                                y_pred_type = y_pred.loc[type_indices]
                            
                            # Calculate metrics
                            metrics = calculate_additional_metrics(y_true_type, y_pred_type)
                            metrics['Vehicle Type'] = vehicle_type
                            metrics['Model'] = model_name
                            metrics['Sample Size'] = len(y_true_type)
                            vehicle_performance.append(metrics)
                
                if vehicle_performance:
                    vehicle_perf_df = pd.DataFrame(vehicle_performance)
                    
                    # Create heatmap of accuracy by vehicle type
                    pivot_accuracy = vehicle_perf_df.pivot(
                        index='Vehicle Type', 
                        columns='Model', 
                        values='Accuracy (%)'
                    )
                    
                    fig = px.imshow(
                        pivot_accuracy.values,
                        x=pivot_accuracy.columns,
                        y=pivot_accuracy.index,
                        color_continuous_scale='RdYlGn',
                        title="Accuracy by Vehicle Type (%)",
                        text_auto=True
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Detailed performance table
                    st.dataframe(vehicle_perf_df.round(2), use_container_width=True)

with tab3:
    st.header("Prediction Analysis & Residuals")
    
    test_data = get_model_predictions()
    
    if test_data:
        y_test = test_data['y_test']
        predictions = test_data['predictions']
        
        # Model selection for detailed analysis
        analysis_model = st.selectbox(
            "Select Model for Detailed Analysis",
            selected_models
        )
        
        if analysis_model in predictions:
            y_pred = predictions[analysis_model]
            
            # Handle NaN values
            if isinstance(y_pred, np.ndarray):
                mask = ~np.isnan(y_pred)
                y_test_clean = y_test[mask]
                y_pred_clean = y_pred[mask]
            else:
                y_test_clean = y_test
                y_pred_clean = y_pred
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Predicted vs Actual scatter plot
                fig = px.scatter(
                    x=y_test_clean,
                    y=y_pred_clean,
                    title=f"{analysis_model}: Predicted vs Actual SoH",
                    labels={'x': 'Actual SoH (%)', 'y': 'Predicted SoH (%)'}
                )
                
                # Add perfect prediction line
                min_val = min(y_test_clean.min(), y_pred_clean.min())
                max_val = max(y_test_clean.max(), y_pred_clean.max())
                fig.add_trace(go.Scatter(
                    x=[min_val, max_val],
                    y=[min_val, max_val],
                    mode='lines',
                    name='Perfect Prediction',
                    line=dict(dash='dash', color='red')
                ))
                
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Residuals plot
                residuals = y_test_clean - y_pred_clean
                
                fig = px.scatter(
                    x=y_pred_clean,
                    y=residuals,
                    title=f"{analysis_model}: Residuals Plot",
                    labels={'x': 'Predicted SoH (%)', 'y': 'Residuals (%)'}
                )
                fig.add_hline(y=0, line_dash="dash", line_color="red")
                st.plotly_chart(fig, use_container_width=True)
            
            # Residuals analysis
            st.subheader("Residuals Analysis")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Mean Residual", f"{residuals.mean():.3f}")
            with col2:
                st.metric("Residual Std", f"{residuals.std():.3f}")
            with col3:
                st.metric("Max Abs Residual", f"{abs(residuals).max():.3f}")
            with col4:
                # Calculate percentage of residuals within ±2%
                within_2_percent = (abs(residuals) <= 2).sum() / len(residuals) * 100
                st.metric("Within ±2%", f"{within_2_percent:.1f}%")
            
            # Distribution of residuals
            col1, col2 = st.columns(2)
            
            with col1:
                fig = px.histogram(
                    x=residuals,
                    nbins=30,
                    title="Distribution of Residuals"
                )
                fig.add_vline(x=0, line_dash="dash", line_color="red")
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Q-Q plot for normality check
                from scipy import stats
                
                qq_data = stats.probplot(residuals, dist="norm")
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=qq_data[0][0],
                    y=qq_data[0][1],
                    mode='markers',
                    name='Residuals'
                ))
                fig.add_trace(go.Scatter(
                    x=qq_data[0][0],
                    y=qq_data[1][1] + qq_data[1][0] * qq_data[0][0],
                    mode='lines',
                    name='Normal Distribution',
                    line=dict(color='red', dash='dash')
                ))
                fig.update_layout(
                    title="Q-Q Plot (Normality Check)",
                    xaxis_title="Theoretical Quantiles",
                    yaxis_title="Sample Quantiles"
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Error distribution by SoH range
            st.subheader("Error Analysis by SoH Range")
            
            # Create SoH bins
            soh_bins = pd.cut(y_test_clean, bins=5, labels=['Very Low', 'Low', 'Medium', 'High', 'Very High'])
            error_by_range = pd.DataFrame({
                'SoH_Range': soh_bins,
                'Absolute_Error': abs(residuals)
            })
            
            fig = px.box(
                error_by_range,
                x='SoH_Range',
                y='Absolute_Error',
                title="Absolute Error by SoH Range"
            )
            st.plotly_chart(fig, use_container_width=True)

with tab4:
    st.header("Feature Importance Comparison")
    
    # Feature importance analysis across models
    if hasattr(ml_models, 'feature_importance') and ml_models.feature_importance:
        
        # Model selection for feature importance
        importance_models = [model for model in selected_models 
                           if model in ml_models.feature_importance]
        
        if importance_models:
            selected_importance_model = st.selectbox(
                "Select Model for Feature Importance",
                importance_models
            )
            
            # Top N features
            n_features = st.slider("Number of Top Features", 5, 30, 15)
            
            # Get feature importance
            importance_df = ml_models.get_feature_importance_top_n(
                selected_importance_model, 
                n_features
            )
            
            if not importance_df.empty:
                col1, col2 = st.columns(2)
                
                with col1:
                    # Feature importance bar chart
                    fig = px.bar(
                        importance_df,
                        x='importance',
                        y='feature',
                        orientation='h',
                        title=f"Top {n_features} Features - {selected_importance_model}",
                        labels={'importance': 'Importance Score', 'feature': 'Feature'}
                    )
                    fig.update_yaxis(categoryorder='total ascending')
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    # Feature importance table
                    st.subheader("Feature Importance Scores")
                    importance_display = importance_df.copy()
                    importance_display['importance'] = importance_display['importance'].round(4)
                    importance_display['rank'] = range(1, len(importance_display) + 1)
                    st.dataframe(
                        importance_display[['rank', 'feature', 'importance']], 
                        use_container_width=True
                    )
                
                # Feature importance comparison across models
                if len(importance_models) > 1:
                    st.subheader("Feature Importance Comparison Across Models")
                    
                    # Get importance for all models
                    all_importance = {}
                    common_features = None
                    
                    for model in importance_models:
                        model_importance = ml_models.get_feature_importance_top_n(model, n_features)
                        if not model_importance.empty:
                            all_importance[model] = dict(zip(
                                model_importance['feature'], 
                                model_importance['importance']
                            ))
                            
                            if common_features is None:
                                common_features = set(model_importance['feature'])
                            else:
                                common_features = common_features.intersection(
                                    set(model_importance['feature'])
                                )
                    
                    if common_features and len(all_importance) > 1:
                        # Create comparison dataframe
                        comparison_importance = pd.DataFrame(all_importance)
                        comparison_importance = comparison_importance.loc[
                            comparison_importance.index.isin(common_features)
                        ].fillna(0)
                        
                        # Heatmap of feature importance
                        fig = px.imshow(
                            comparison_importance.values,
                            x=comparison_importance.columns,
                            y=comparison_importance.index,
                            color_continuous_scale='Blues',
                            title="Feature Importance Heatmap Across Models"
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Feature ranking comparison
                        ranking_comparison = pd.DataFrame()
                        for model in importance_models:
                            if model in all_importance:
                                model_rank = pd.Series(all_importance[model]).rank(
                                    ascending=False, method='min'
                                )
                                ranking_comparison[f'{model}_rank'] = model_rank
                        
                        ranking_comparison = ranking_comparison.loc[common_features]
                        st.dataframe(ranking_comparison, use_container_width=True)
                
                # Feature categories analysis
                st.subheader("Feature Categories Analysis")
                
                if 'training_features' in st.session_state:
                    features = importance_df['feature'].tolist()
                    
                    # Categorize features
                    categories = {
                        'Original': [f for f in features if not any(x in f for x in ['rolling', 'lag', '_interaction', '_normalized', '_zscore', 'rate_of_change'])],
                        'Rolling Statistics': [f for f in features if 'rolling' in f],
                        'Lag Features': [f for f in features if 'lag' in f],
                        'Interaction Features': [f for f in features if '_interaction' in f or '_ratio' in f],
                        'Derived Features': [f for f in features if any(x in f for x in ['normalized', 'zscore', 'rate_of_change', 'cumulative'])]
                    }
                    
                    # Calculate category importance
                    category_importance = {}
                    for category, category_features in categories.items():
                        category_features_in_top = [f for f in category_features if f in features]
                        if category_features_in_top:
                            total_importance = importance_df[
                                importance_df['feature'].isin(category_features_in_top)
                            ]['importance'].sum()
                            category_importance[category] = total_importance
                    
                    if category_importance:
                        fig = px.pie(
                            values=list(category_importance.values()),
                            names=list(category_importance.keys()),
                            title=f"Feature Category Importance - {selected_importance_model}"
                        )
                        st.plotly_chart(fig, use_container_width=True)
        
        else:
            st.info("Feature importance analysis is available for Random Forest and XGBoost models.")
    
    else:
        st.warning("No feature importance data available. Train Random Forest or XGBoost models first.")

with tab5:
    st.header("Model Selection Guide & Recommendations")
    
    # Model selection criteria
    st.subheader("Model Selection Criteria")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Performance Criteria:**")
        st.write("✅ Accuracy > 95% (primary target)")
        st.write("✅ Low prediction error (MAE < 3%)")  
        st.write("✅ High R² score (> 0.90)")
        st.write("✅ Consistent performance across vehicle types")
        st.write("✅ Robust to outliers and missing data")
    
    with col2:
        st.write("**Practical Criteria:**")
        st.write("⚡ Fast prediction time")
        st.write("💾 Reasonable model size")
        st.write("🔍 Model interpretability")
        st.write("🔧 Easy to retrain and update")
        st.write("📊 Good confidence interval estimation")
    
    # Model recommendations based on results
    comparison_df = ml_models.get_model_comparison()
    
    if not comparison_df.empty:
        st.subheader("Recommendation Engine")
        
        # Score each model based on multiple criteria
        recommendation_scores = []
        
        for _, model_row in comparison_df.iterrows():
            model_name = model_row['Model']
            
            # Performance score (0-100)
            accuracy_score = min(100, model_row['Accuracy (%)'])
            r2_score_norm = model_row['Test R²'] * 100
            mae_score = max(0, 100 - (model_row['MAE'] * 10))  # Penalize high MAE
            
            performance_score = (accuracy_score * 0.4 + r2_score_norm * 0.3 + mae_score * 0.3)
            
            # Practical score (simplified scoring)
            if model_name == 'Random Forest':
                practical_score = 85  # High interpretability, medium speed
            elif model_name == 'XGBoost':
                practical_score = 90  # Good balance of performance and speed
            elif model_name == 'LSTM':
                practical_score = 70  # High performance but complex
            else:
                practical_score = 75  # Default
            
            # Overall score
            overall_score = (performance_score * 0.7 + practical_score * 0.3)
            
            recommendation_scores.append({
                'Model': model_name,
                'Performance Score': performance_score,
                'Practical Score': practical_score,
                'Overall Score': overall_score,
                'Accuracy': model_row['Accuracy (%)'],
                'Meets Target': model_row['Accuracy (%)'] > 95
            })
        
        recommendation_df = pd.DataFrame(recommendation_scores)
        recommendation_df = recommendation_df.sort_values('Overall Score', ascending=False)
        
        # Display recommendations
        st.dataframe(recommendation_df.round(1), use_container_width=True)
        
        # Top recommendation
        top_model = recommendation_df.iloc[0]
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.success(f"""
            **🏆 Recommended Model**
            
            **{top_model['Model']}**
            
            Overall Score: {top_model['Overall Score']:.1f}/100
            Accuracy: {top_model['Accuracy']:.1f}%
            Target Status: {'✅ Met' if top_model['Meets Target'] else '❌ Not Met'}
            """)
        
        with col2:
            # Model characteristics
            if top_model['Model'] == 'Random Forest':
                characteristics = """
                **Characteristics:**
                - High interpretability
                - Robust to outliers
                - Good feature importance
                - Stable predictions
                - Easy to tune
                """
            elif top_model['Model'] == 'XGBoost':
                characteristics = """
                **Characteristics:**
                - Excellent performance
                - Fast training/prediction
                - Built-in regularization
                - Handles missing data
                - Good for production
                """
            elif top_model['Model'] == 'LSTM':
                characteristics = """
                **Characteristics:**
                - Captures time patterns
                - Complex relationships
                - Sequential dependencies
                - Requires more data
                - Less interpretable
                """
            else:
                characteristics = "**Standard ML model**"
            
            st.info(characteristics)
        
        with col3:
            # Implementation recommendations
            if top_model['Meets Target']:
                implementation = """
                **✅ Ready for Production**
                
                - Deploy immediately
                - Set up monitoring
                - Plan regular retraining
                - Implement alerting
                """
            else:
                implementation = """
                **⚠️ Needs Improvement**
                
                - Collect more data
                - Try ensemble methods
                - Feature engineering
                - Hyperparameter tuning
                """
            
            st.warning(implementation)
    
    # Deployment considerations
    st.subheader("Deployment Considerations")
    
    deployment_guide = {
        "Random Forest": {
            "Pros": ["High interpretability", "Robust performance", "Fast predictions", "Easy maintenance"],
            "Cons": ["May overfit with small datasets", "Less accurate than ensemble methods"],
            "Best For": ["Interpretable predictions", "Regulatory compliance", "Feature analysis"],
            "Production Tips": ["Use 100-300 trees", "Monitor feature drift", "Regular retraining monthly"]
        },
        "XGBoost": {
            "Pros": ["Excellent accuracy", "Built-in regularization", "Handles missing data", "Fast"],
            "Cons": ["Less interpretable", "More hyperparameters", "Can overfit"],
            "Best For": ["High-accuracy requirements", "Production systems", "Large datasets"],
            "Production Tips": ["Use early stopping", "Monitor for overfitting", "GPU acceleration available"]
        },
        "LSTM": {
            "Pros": ["Captures temporal patterns", "Complex relationships", "Sequential modeling"],
            "Cons": ["Requires more data", "Complex to tune", "Computationally intensive"],
            "Best For": ["Time series patterns", "Long-term forecasting", "Complex sequences"],
            "Production Tips": ["Use GPU acceleration", "Batch predictions", "Monitor model drift"]
        }
    }
    
    for model_name, guide in deployment_guide.items():
        if model_name in selected_models:
            with st.expander(f"{model_name} Deployment Guide"):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Pros:**")
                    for pro in guide["Pros"]:
                        st.write(f"✅ {pro}")
                    
                    st.write("**Cons:**")
                    for con in guide["Cons"]:
                        st.write(f"❌ {con}")
                
                with col2:
                    st.write("**Best For:**")
                    for use_case in guide["Best For"]:
                        st.write(f"🎯 {use_case}")
                    
                    st.write("**Production Tips:**")
                    for tip in guide["Production Tips"]:
                        st.write(f"💡 {tip}")
    
    # Final recommendation summary
    st.subheader("Executive Summary")
    
    if not comparison_df.empty:
        models_meeting_target = comparison_df[comparison_df['Accuracy (%)'] > 95]
        
        if len(models_meeting_target) > 0:
            best_model = models_meeting_target.iloc[0]['Model']
            best_accuracy = models_meeting_target.iloc[0]['Accuracy (%)']
            
            st.success(f"""
            **🎉 SUCCESS: Target Achieved!**
            
            The **{best_model}** model has achieved {best_accuracy:.1f}% accuracy, exceeding the 95% target.
            
            **Next Steps:**
            1. Deploy {best_model} for production use
            2. Implement real-time monitoring
            3. Set up automated retraining pipeline
            4. Create alerting for model performance degradation
            5. Plan A/B testing with alternative models
            
            **Confidence Level:** High - Ready for immediate deployment
            """)
        else:
            best_model = comparison_df.iloc[0]['Model']
            best_accuracy = comparison_df.iloc[0]['Accuracy (%)']
            
            st.warning(f"""
            **⚠️ TARGET NOT MET**
            
            Best model ({best_model}) achieved {best_accuracy:.1f}% accuracy, below the 95% target.
            
            **Improvement Strategies:**
            1. Collect more diverse training data
            2. Advanced feature engineering
            3. Ensemble methods (combining models)
            4. Hyperparameter optimization
            5. Data quality improvements
            
            **Recommendation:** Continue development before production deployment
            """)

# Footer
st.markdown("---")
st.markdown("""
**Model Comparison System** - Comprehensive evaluation and selection of ML algorithms for EV battery SoH prediction.
Use this analysis to select the optimal model for your production deployment.
""")

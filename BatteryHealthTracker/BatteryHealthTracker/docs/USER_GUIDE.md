# User Guide: EV Battery State-of-Health Prognostics System

## Table of Contents
1. [Getting Started](#getting-started)
2. [Navigation Overview](#navigation-overview)
3. [Data Ingestion](#data-ingestion)
4. [Model Training](#model-training)
5. [Real-Time Prediction](#real-time-prediction)
6. [Fleet Dashboard](#fleet-dashboard)
7. [Model Comparison](#model-comparison)
8. [Troubleshooting](#troubleshooting)

---

## 1. Getting Started

### System Requirements
- Web browser (Chrome, Firefox, Safari, Edge)
- Internet connection for initial setup
- No software installation required (web-based application)

### First Time Access
1. Navigate to the application URL (typically http://localhost:5000 for local deployment)
2. The main dashboard loads automatically
3. Check the sidebar for "Database Status: Connected ✓"
4. If not connected, contact your system administrator

### Understanding the Interface
- **Main Content Area**: Displays current page content and visualizations
- **Sidebar**: Navigation menu, filters, and system status
- **Top Navigation**: Page tabs for different system functions
- **Status Indicators**: Real-time system health and database connection status

---

## 2. Navigation Overview

The system is organized into five main sections accessible via the sidebar:

### 🏠 Main Dashboard
- System overview and quick statistics
- Fleet composition summary
- Navigation guide to other sections
- Database connectivity status

### 📊 Data Ingestion
- Generate synthetic fleet data
- Configure vehicle types and parameters
- Load existing data from database
- Data quality assessment tools

### 🤖 Model Training  
- Advanced feature engineering pipeline
- Train multiple machine learning models
- Hyperparameter optimization
- Performance evaluation

### 🔮 Real-Time Prediction
- Live battery health forecasting
- Individual vehicle analysis
- Confidence interval calculation
- Degradation trend visualization

### 🚗 Fleet Dashboard
- Multi-vehicle monitoring
- Fleet-wide analytics
- Filtering and search capabilities
- Maintenance alerts and insights

### 📈 Model Comparison
- Performance metrics comparison
- Feature importance analysis
- Model validation results
- Export capabilities

---

## 3. Data Ingestion

### Purpose
Generate or load battery telemetry data for analysis and model training.

### Step-by-Step Guide

#### 3.1 Data Generation
1. **Access the Page**: Click "Data Ingestion" in the sidebar
2. **Configure Fleet Settings**:
   - **Number of Vehicles**: Use slider to select 10-100 vehicles
   - **Data Collection Days**: Choose 30-365 days of historical data
   
3. **Set Vehicle Distribution**:
   - **Cars**: Percentage of cars in fleet (default: 40%)
   - **Buses**: Percentage of buses in fleet (default: 30%) 
   - **Trucks**: Percentage of trucks in fleet (default: 20%)
   - **Motorcycles**: Percentage of motorcycles in fleet (default: 10%)
   - Ensure total equals 100%

4. **Data Quality Options**:
   - **Add Missing Values**: Simulate real-world data gaps (0-10%)
   - **Add Outliers**: Include anomalous readings (0-5%)

5. **Generate Data**: Click "🚀 Generate Fleet Data"
   - Progress bar shows generation status
   - Automatic save to database if connected
   - Success message confirms completion

#### 3.2 Loading Existing Data
1. **Check Database Status**: Ensure "Connected ✓" status
2. **Load Data**: Click "📂 Load Existing Data from Database"
3. **Verification**: Review loaded records count and vehicle information

#### 3.3 Data Quality Assessment
1. **Navigate to "Data Quality" Tab**
2. **Run Quality Check**: Click "🔍 Run Data Quality Check"
3. **Review Results**:
   - Missing value percentages
   - Outlier detection results
   - Data completeness scores
   - Duplicate record counts

4. **Clean Data**: Click "🧹 Clean Data" if issues detected
   - Automatic interpolation of missing values
   - Outlier handling using statistical methods
   - Duplicate removal

#### 3.3 Data Overview
1. **Sample Preview**: Review first 10 rows of generated data
2. **Statistics**: Check numerical summaries (mean, std, min, max)
3. **Visualizations**: Examine vehicle type distribution charts
4. **Export Options**: Download data in various formats

### Best Practices
- Start with 30-50 vehicles for initial testing
- Use 90-180 days of data for robust model training
- Enable data quality issues for realistic training scenarios
- Always verify data completeness before model training

---

## 4. Model Training

### Purpose
Train machine learning models to predict battery State-of-Health with high accuracy.

### Prerequisites
- Fleet data must be generated or loaded (see Data Ingestion)
- Minimum 1000 data points recommended
- Multiple vehicle types preferred for robust training

### Step-by-Step Guide

#### 4.1 Feature Engineering
1. **Access Model Training Page**
2. **Configure Feature Engineering**:
   - **Rolling Statistics**: Enable for time-based features (recommended)
   - **Interaction Features**: Enable for feature combinations
   - **Lag Features**: Enable for time series dependencies (optional)

3. **Start Feature Engineering**: Click "🔧 Engineer Features"
   - Progress tracking shows current step
   - New features created from raw telemetry data
   - Feature count increases from ~15 to 50+ features

4. **Review Results**:
   - Compare original vs. engineered feature counts
   - Examine feature categories (time-based, statistical, derived)
   - Preview sample of engineered features

#### 4.2 Training Configuration
1. **Model Selection**:
   - **Random Forest**: Enable for ensemble learning (recommended)
   - **XGBoost**: Enable for gradient boosting (recommended) 
   - **LSTM Neural Network**: Enable for deep learning (optional)

2. **Data Preprocessing**:
   - **Test Set Size**: 10-40% (default: 20%)
   - **Feature Selection**: Enable to reduce overfitting
   - **Max Features**: 20-100 features (default: 50)

3. **Training Options**:
   - **Cross Validation**: Enable for robust evaluation
   - **Hyperparameter Tuning**: Enable for optimal performance

#### 4.3 Model Training Process
1. **Start Training**: Click "🚀 Start Training"
2. **Monitor Progress**:
   - Feature preparation phase
   - Feature selection phase  
   - Model training phase (may take several minutes)
   - Model evaluation phase
   - Database saving phase

3. **Review Results**:
   - Training completion confirmation
   - Performance metrics for each model
   - Automatic model saving to database

#### 4.4 Training Results Analysis
1. **Performance Metrics**:
   - **Accuracy Percentage**: Target >95%
   - **R² Score**: Closer to 1.0 is better
   - **MAE (Mean Absolute Error)**: Lower is better
   - **RMSE (Root Mean Square Error)**: Lower is better

2. **Model Comparison**:
   - Compare performance across all trained models
   - Identify best-performing model for predictions
   - Note training vs. testing performance

3. **Feature Importance**:
   - View top 20 most important features
   - Understand which factors drive predictions
   - Verify feature engineering effectiveness

#### 4.5 Model Validation
1. **Cross-Validation Results**:
   - Review k-fold validation scores
   - Check for overfitting indicators
   - Assess model stability across data splits

2. **Residual Analysis**:
   - Examine prediction errors
   - Check for systematic biases
   - Validate model assumptions

### Troubleshooting Training Issues
- **Low Accuracy (<90%)**: Increase data size or feature engineering
- **Training Takes Too Long**: Reduce vehicle count or feature count
- **Memory Errors**: Reduce batch size or enable data chunking
- **Convergence Issues**: Try different hyperparameters or models

---

## 5. Real-Time Prediction

### Purpose
Generate live State-of-Health predictions for individual vehicles with confidence intervals.

### Prerequisites  
- Trained models available (see Model Training)
- Fleet data with recent vehicle information
- Database connection for prediction logging

### Step-by-Step Guide

#### 5.1 Prediction Setup
1. **Access Real-Time Prediction Page**
2. **Model Selection**:
   - Choose from available trained models
   - XGBoost typically provides highest accuracy
   - Random Forest offers good interpretability

3. **Vehicle Selection**:
   - Select specific vehicle from dropdown
   - Review vehicle information (type, age, capacity)
   - Check last data update timestamp

4. **Prediction Parameters**:
   - **Prediction Horizon**: 1-365 days ahead
   - **Confidence Level**: 80-99% (default: 95%)

#### 5.2 Making Predictions
1. **Single Prediction**: Click "🔮 Predict SoH"
   - Instant prediction generation
   - Confidence interval calculation
   - Automatic database logging

2. **Real-Time Updates**: Enable "Real-Time Updates"
   - Automatic prediction refresh every 1-10 seconds
   - Continuous monitoring capability
   - Live dashboard updates

3. **Prediction Results**:
   - **Current SoH**: Latest measured State-of-Health
   - **Predicted SoH**: Model forecast
   - **Confidence Range**: Upper and lower bounds
   - **Prediction Accuracy**: Historical accuracy for this model

#### 5.3 Trend Analysis
1. **Historical Trends**:
   - View SoH degradation over time
   - Compare multiple vehicles
   - Identify acceleration in degradation

2. **Future Projections**:
   - 30-day forecast with confidence bands
   - Maintenance threshold indicators
   - Replacement timeline estimates

#### 5.4 Alert Configuration
1. **Enable Alerts**: Toggle alert functionality
2. **Warning Threshold**: Set SoH percentage (default: 85%)
3. **Critical Threshold**: Set urgent action level (default: 75%)
4. **Alert Actions**:
   - Visual notifications in interface
   - Automatic logging to database
   - Integration with maintenance systems

### Interpreting Results
- **High Confidence (narrow bands)**: Model is certain about prediction
- **Low Confidence (wide bands)**: Higher uncertainty, consider additional data
- **Rapid Degradation**: SoH dropping >1% per month may indicate issues
- **Stable Performance**: SoH changes <0.5% per month are normal

---

## 6. Fleet Dashboard

### Purpose
Monitor entire vehicle fleets with comprehensive analytics and filtering capabilities.

### Features Overview
- Multi-vehicle health monitoring
- Fleet-wide performance analytics
- Advanced filtering and search
- Maintenance planning support
- Export capabilities for reporting

### Step-by-Step Guide

#### 6.1 Fleet Overview
1. **Access Fleet Dashboard Page**
2. **Quick Statistics Review**:
   - Total vehicles in fleet
   - Average fleet SoH
   - Vehicles requiring attention
   - Recent prediction count

3. **Fleet Composition**:
   - Vehicle type distribution (pie chart)
   - Age distribution histogram
   - SoH distribution by vehicle type

#### 6.2 Filtering and Search
1. **Vehicle Type Filter**:
   - Select specific vehicle types
   - Multi-select for comparison
   - Real-time chart updates

2. **SoH Range Filter**:
   - Set minimum and maximum SoH values
   - Focus on vehicles needing attention
   - Exclude healthy vehicles from view

3. **Age Filter**:
   - Filter by vehicle age in days
   - Analyze age-related degradation
   - Compare new vs. older vehicles

4. **Time Range Selection**:
   - Select specific date ranges
   - Focus on recent data
   - Historical trend analysis

#### 6.3 Fleet Analytics
1. **Health Distribution Analysis**:
   - SoH histogram across fleet
   - Identify distribution patterns
   - Spot outliers and anomalies

2. **Performance Trends**:
   - Fleet-wide degradation trends
   - Seasonal pattern identification
   - Performance comparison by type

3. **Maintenance Insights**:
   - Vehicles approaching thresholds
   - Predicted maintenance schedules
   - Cost optimization recommendations

#### 6.4 Individual Vehicle Deep Dive
1. **Vehicle Selection**: Click on individual vehicles in charts
2. **Detailed Analysis**:
   - Complete degradation history
   - Recent prediction accuracy
   - Maintenance recommendations
   - Performance compared to fleet average

3. **Prediction History**:
   - Historical prediction accuracy
   - Confidence interval evolution
   - Model performance over time

#### 6.5 Alerts and Notifications
1. **Fleet-Wide Alerts**:
   - Vehicles below warning thresholds
   - Rapid degradation alerts
   - Maintenance due notifications

2. **Alert Management**:
   - Acknowledge alerts
   - Set custom thresholds
   - Configure notification preferences

### Best Practices
- Review fleet dashboard daily for proactive maintenance
- Use filters to focus on problematic vehicle segments
- Export reports for maintenance team coordination
- Track alert trends to identify systemic issues

---

## 7. Model Comparison

### Purpose
Analyze and compare performance of different machine learning models for optimal selection.

### Available Comparisons
- Model accuracy metrics
- Feature importance analysis
- Performance across vehicle types
- Prediction confidence evaluation

### Step-by-Step Guide

#### 7.1 Performance Metrics Comparison
1. **Access Model Comparison Page**
2. **Select Models**: Choose models to compare
3. **Choose Metrics**:
   - **Accuracy Percentage**: Overall prediction accuracy
   - **R² Score**: Explained variance measure
   - **MAE**: Mean Absolute Error
   - **RMSE**: Root Mean Square Error

4. **Sorting Options**:
   - Sort by accuracy (highest first)
   - Sort by R² score
   - Sort by error metrics (lowest first)

5. **Visualization**:
   - Bar charts comparing model performance
   - Side-by-side metric tables
   - Performance radar charts

#### 7.2 Feature Importance Analysis
1. **Model Selection**: Choose specific model for analysis
2. **Feature Importance Views**:
   - Top 20 most important features
   - Feature category breakdown
   - Importance score distributions

3. **Interpretation**:
   - Understand prediction drivers
   - Validate feature engineering choices
   - Identify irrelevant features

#### 7.3 Vehicle Type Performance
1. **Type-Specific Analysis**:
   - Model performance by vehicle type (car, bus, truck, motorcycle)
   - Identify models that work best for specific types
   - Understand type-specific challenges

2. **Cross-Type Validation**:
   - Model generalization across types
   - Transfer learning effectiveness
   - Specialized vs. general models

#### 7.4 Prediction Confidence Analysis
1. **Confidence Metrics**:
   - Average confidence interval width
   - Confidence calibration accuracy
   - Over/under-confidence identification

2. **Reliability Assessment**:
   - Models with most reliable uncertainty estimates
   - Confidence vs. accuracy relationships
   - Risk assessment capabilities

#### 7.5 Model Selection Recommendations
1. **Best Overall Model**: Highest accuracy across all metrics
2. **Most Reliable Model**: Best uncertainty quantification  
3. **Fastest Model**: Shortest prediction time
4. **Most Interpretable**: Clearest feature importance

### Export and Reporting
1. **Performance Reports**: Export comparison tables and charts
2. **Feature Analysis**: Export feature importance data
3. **Recommendation Summary**: Generated model selection guidance

---

## 8. Troubleshooting

### Common Issues and Solutions

#### Database Connection Issues
**Symptoms**: "Database Status: Not Connected ❌" in sidebar
**Solutions**:
1. Check internet connectivity
2. Verify database server is running
3. Contact system administrator
4. Check environment variables are set correctly

#### Data Generation Failures
**Symptoms**: Error messages during data generation
**Solutions**:
1. Reduce number of vehicles or days
2. Check available memory and disk space
3. Restart the application
4. Try smaller batches of data

#### Model Training Errors
**Symptoms**: Training fails or produces low accuracy
**Solutions**:
1. Ensure sufficient data (minimum 1000 records)
2. Check for data quality issues
3. Enable feature engineering
4. Try different model combinations
5. Increase test set size

#### Slow Performance
**Symptoms**: Pages load slowly or operations time out
**Solutions**:
1. Reduce data size for analysis
2. Use data filters to limit scope
3. Clear browser cache
4. Check system resources
5. Contact administrator for performance optimization

#### Prediction Errors
**Symptoms**: Prediction requests fail or return unrealistic values
**Solutions**:
1. Verify models are properly trained
2. Check input data quality
3. Ensure vehicle has recent data
4. Try different prediction models
5. Retrain models if consistently inaccurate

### Getting Help

#### Built-in Help
- Hover over any ⓘ icon for contextual help
- Check status messages in the sidebar
- Review error messages for specific guidance

#### Documentation
- Technical Documentation: Detailed model equations and algorithms
- API Reference: Complete function and class documentation
- This User Guide: Step-by-step usage instructions

#### Support Contacts
- System Administrator: Technical infrastructure issues
- Data Team: Data quality and modeling questions  
- Product Team: Feature requests and functionality questions

### Best Practices for Smooth Operation
1. **Regular Data Refresh**: Generate new data weekly for up-to-date models
2. **Model Retraining**: Retrain models monthly or when accuracy drops
3. **Database Maintenance**: Allow administrators to perform regular maintenance
4. **Browser Management**: Use modern browsers and clear cache regularly
5. **Data Backup**: Export important analysis results for offline storage

---

This user guide provides comprehensive instructions for using the EV Battery State-of-Health Prognostics System. Follow the step-by-step procedures for optimal results and refer to the troubleshooting section when issues arise.
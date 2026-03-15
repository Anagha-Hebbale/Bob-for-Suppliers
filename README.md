# Bob-the-Suppliers
Bob-for-Suppliers is an integrated supply chain intelligence system. The platform leverages data analytics and machine learning to help businesses improve operational efficiency, reduce risks, and optimize their supply chain processes through an interactive Streamlit dashboard.

🎯 Key Features
 1. **Executive Dashboard**
- Real-time KPIs monitoring
- Late delivery rate tracking
- Average profit per order analysis
- Market-wise performance metrics
- Interactive visualizations

 2. **Performance Monitoring**
- Delivery status distribution
- Shipping mode analysis
- Category-wise profit breakdown
- Market-wise risk assessment
- Regional performance tracking

 3. **Anomaly & Fraud Detection**
- Interactive anomaly detection with adjustable sensitivity
- Z-score based outlier identification
- Multi-dimensional analysis
- Visual anomaly highlighting
- Custom threshold settings

4. **Predictive Risk Modeling**
- Late delivery risk prediction
- Market-specific risk factors
- Shipping mode risk assessment
- Real-time risk scoring
- Visual gauge indicators (High/Medium/Low)

5. **Seasonal Trend Analysis**
- Monthly order volume trends
- Day-of-week patterns
- Peak season identification
- Quarter-wise analysis
- Year-over-year comparisons

6. **Intelligent Recommendations**
- Automated improvement suggestions
- Impact vs Effort analysis
- Priority matrix visualization
- Quick wins identification
- Actionable business insights

🏗️ Project Structure

```
Bob-for-Suppliers/
│
├── dashboard/
|   |-app.py                         
│
├── data/
│   ├── cleaned_data.csv            # Preprocessed supply chain data
│   ├── DataCoSupplyChainDataset.csv # Raw dataset
│   ├── DescriptionDataCoSupplyChain.csv # Data dictionary
│   ├── final_operational_insights.csv # Generated insights
│   ├── late_delivery_feature_importance.csv # Feature importance
│   ├── operational_insights.csv    # Operational metrics
│   ├── processed_data.csv           # Feature-engineered data
│   ├── profit_feature_importance.csv # Profit drivers
│   └── tokenized_access_logs.csv    # Access logs
│
├── models/
│   ├── abs_loss_model.pkl           # Absolute loss predictor
│   ├── encoders.pkl                  # Label encoders
│   ├── late_delivery_model.pkl       # Primary delay predictor
│   ├── late_delivery_model1.pkl      # Secondary delay predictor
│   ├── margin_loss_model.pkl         # Margin loss calculator
│   ├── profit_prediction_model.pkl    # Profit forecaster
│   └── return_risk_model.pkl          # Return probability model
│
├── audit_logic.py                    # Audit trail and validation
├── predict_risk.py                    # Risk prediction engine
├── train_models.py                     # Model training pipeline
│
├── src/
│   ├── accuracy_graphs.py              # Accuracy visualizations
│   ├── anomaly.py                      # Anomaly detection algorithms
│   ├── data_preprocessing.py            # Data cleaning
│   ├── debug.py                         # Debugging utilities
│   ├── evaluate_models.py                # Model evaluation
│   ├── feature_engineering.py             # Feature creation
│   ├── feature_importance.py               # Driver analysis
│   ├── generate_insights.py                 # Business insights
│   ├── model_visualizations.py               # Performance charts
│   ├── seasonal_demand_analysis.py            # Demand patterns
│   ├── shipping_cost_analysis.py               # Cost optimization
│   ├── train_late_delivery_model.py             # Delay training
│   ├── train_profit_model.py                     # Profit training
│   ├── train_return_model.py                      # Return training
│    
├── requirements.txt                    # Dependencies
├── requirements.txt.txt                 # Backup requirements

```

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- pip package manager
- Virtual environment (recommended)

### Dataset Link:
https://www.kaggle.com/datasets/shashwatwork/dataco-smart-supply-chain-for-big-data-analysis

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/Bob-for-Suppliers.git
cd Bob-for-Suppliers
```

2. **Create and activate virtual environment**
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Run the codes to train model**

5. **Launch the dashboard**
```bash
streamlit run app.py
```

## 💻 Dashboard Pages

### 📊 **Executive Dashboard**
- High-level KPIs and metrics
- Market-wise profit analysis
- Order distribution visualization
- Bob's daily insights
- Quick overview of key performance indicators

### 📈 **Performance Metrics**
- **Delivery Performance**: Status distribution and risk analysis
- **Shipping Analysis**: Mode distribution and delay statistics
- **Category Analysis**: Top-performing product categories

### 🔍 **Anomaly Detection**
- Interactive scatter plot visualization
- Adjustable sensitivity threshold (2.0-4.0)
- Real-time anomaly identification
- Multi-feature analysis
- Color-coded anomaly highlighting

### 🎯 **Predictive Risk Model**
- Form-based input for predictions
- Market, shipping mode, and category selection
- Real-time risk calculation
- Visual gauge indicator
- Risk level classification (High/Medium/Low)

### 📅 **Seasonal Trends**
- Monthly order volume trends
- Day-of-week patterns
- Peak season identification
- Average line for comparison
- Interactive time series plots

## 📊 Key Metrics Tracked

- **Late Delivery Rate**: Real-time tracking with status indicators
- **Average Profit per Order**: Financial performance monitoring
- **Total Orders**: Volume tracking
- **Active Markets**: Market presence analysis
- **Delivery Status**: On-time vs late distribution
- **Shipping Delays**: Statistical analysis

## 🔧 Configuration

### Data Requirements
The dashboard expects a CSV file at `data/cleaned_data.csv` with the following columns:
- `Market` - Geographic market
- `Shipping Mode` - Shipping method
- `Delivery Status` - On-time/Late status
- `Order Profit Per Order` - Profit metrics
- `Category Name` - Product categories
- `Late_delivery_risk` - Risk scores
- `order date (DateOrders)` - Temporal data

### Custom Styling
The dashboard includes custom CSS for:
- Responsive metric cards
- Color-coded insight boxes
- Bob themed quotes
- Professional typography
- Interactive elements

## 📈 Sample Insights Generated

### Market Strategy
```
Consider reducing operations in [worst_market] market and focusing more on [best_market] 
where profits are [X]x higher.
```

### Shipping Optimization
```
'[shipping_mode]' has the highest late delivery risk ([X]%). 
Consider renegotiating contracts or finding alternatives.
```

### Customer Targeting
```
While '[top_segment]' has most orders, '[best_segment]' is most profitable. 
Create loyalty program for [best_segment] segment.
```

### Inventory Planning
```
Peak months are [months]. Increase safety stock by 30% during these months 
to prevent stockouts.
```





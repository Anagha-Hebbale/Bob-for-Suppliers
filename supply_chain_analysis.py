import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# =============================================================================
# 1. LOAD AND PREPARE DATA
# =============================================================================

def load_and_prepare_data(file_path='DataCoSupplyChainDataset.csv'):
    """Load and prepare the supply chain dataset"""
    print("Loading data...")
    
    # Load data
    df = pd.read_csv(file_path, encoding='latin1')
    
    # Convert date columns
    df['order date (DateOrders)'] = pd.to_datetime(df['order date (DateOrders)'])
    
    # Create derived columns
    df['Year'] = df['order date (DateOrders)'].dt.year
    df['Month'] = df['order date (DateOrders)'].dt.month
    df['Year-Month'] = df['order date (DateOrders)'].dt.to_period('M').astype(str)
    
    # Create On-Time Delivery flag (1 if on time, 0 if late)
    df['On_Time'] = (df['Delivery Status'] == 'Advance shipping').astype(int)
    
    print(f"Loaded {len(df):,} records")
    return df

# =============================================================================
# 2. LOGISTICS PERFORMANCE METRICS
# =============================================================================

def calculate_logistics_metrics(df):
    """Calculate key logistics performance metrics"""
    
    print("\n" + "="*50)
    print("LOGISTICS PERFORMANCE METRICS")
    print("="*50)
    
    # Overall metrics
    total_orders = len(df)
    on_time_pct = df['On_Time'].mean() * 100
    avg_shipping_days = df['Days for shipping (real)'].mean()
    late_risk_pct = df['Late_delivery_risk'].mean() * 100
    
    print(f"\n📊 KEY PERFORMANCE INDICATORS:")
    print(f"   Total Orders: {total_orders:,}")
    print(f"   On-Time Delivery %: {on_time_pct:.1f}%")
    print(f"   Average Shipping Days: {avg_shipping_days:.1f}")
    print(f"   Late Delivery Risk %: {late_risk_pct:.1f}%")
    
    # Shipping mode performance
    print(f"\n📦 SHIPPING MODE PERFORMANCE:")
    shipping_perf = df.groupby('Shipping Mode').agg({
        'Days for shipping (real)': ['mean', 'std'],
        'Late_delivery_risk': 'mean',
        'Order Id': 'count'
    }).round(2)
    
    shipping_perf.columns = ['Avg Days', 'Std Days', 'Late Risk %', 'Order Count']
    shipping_perf['Late Risk %'] = shipping_perf['Late Risk %'] * 100
    print(shipping_perf)
    
    # Delivery status distribution
    print(f"\n📋 DELIVERY STATUS DISTRIBUTION:")
    status_dist = df['Delivery Status'].value_counts()
    status_pct = df['Delivery Status'].value_counts(normalize=True) * 100
    status_df = pd.DataFrame({
        'Count': status_dist,
        'Percentage': status_pct.round(1)
    })
    print(status_df)
    
    return shipping_perf, status_df

# =============================================================================
# 3. RISK HEAT MAP CALCULATIONS
# =============================================================================

def calculate_risk_metrics(df):
    """Calculate risk metrics for heatmaps"""
    
    print("\n" + "="*50)
    print("RISK HEAT MAP METRICS")
    print("="*50)
    
    # Market risk profile
    print(f"\n🌍 MARKET RISK PROFILE (% Late Delivery):")
    market_risk = df.groupby('Market')['Late_delivery_risk'].mean() * 100
    market_risk = market_risk.sort_values(ascending=False)
    for market, risk in market_risk.items():
        print(f"   {market}: {risk:.1f}%")
    
    # Category risk profile
    print(f"\n📦 PRODUCT CATEGORY RISK PROFILE (% Late Delivery):")
    category_risk = df.groupby('Category Name')['Late_delivery_risk'].mean() * 100
    category_risk = category_risk.sort_values(ascending=False)
    for category, risk in category_risk.items():
        print(f"   {category}: {risk:.1f}%")
    
    # Market × Category risk matrix
    print(f"\n📊 MARKET × CATEGORY RISK MATRIX (% Late Delivery):")
    risk_matrix = pd.pivot_table(
        df,
        values='Late_delivery_risk',
        index='Market',
        columns='Category Name',
        aggfunc='mean'
    ) * 100
    
    print(risk_matrix.round(1))
    
    # Order status by market
    print(f"\n📋 ORDER STATUS BY MARKET:")
    status_by_market = pd.crosstab(
        df['Market'],
        df['Order Status'],
        values=df['Order Id'],
        aggfunc='count',
        normalize='index'
    ) * 100
    
    print(status_by_market.round(1))
    
    return market_risk, category_risk, risk_matrix, status_by_market

# =============================================================================
# 4. CREATE VISUALIZATIONS
# =============================================================================

def create_logistics_visualizations(df):
    """Create logistics performance charts"""
    
    print("\n" + "="*50)
    print("CREATING LOGISTICS VISUALIZATIONS")
    print("="*50)
    
    # 1. Shipping Mode Performance Bar Chart
    shipping_perf = df.groupby('Shipping Mode').agg({
        'Days for shipping (real)': 'mean',
        'Late_delivery_risk': 'mean'
    }).round(2)
    
    fig1 = px.bar(
        shipping_perf.reset_index(),
        x='Shipping Mode',
        y='Days for shipping (real)',
        color='Late_delivery_risk',
        title='Average Shipping Days by Mode',
        labels={'Days for shipping (real)': 'Average Days', 'Late_delivery_risk': 'Late Risk'},
        color_continuous_scale='RdYlGn_r'
    )
    fig1.show()
    
    # 2. Monthly Performance Trends
    monthly_perf = df.groupby('Year-Month').agg({
        'On_Time': 'mean',
        'Late_delivery_risk': 'mean'
    }).reset_index()
    
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(
        x=monthly_perf['Year-Month'],
        y=monthly_perf['On_Time'] * 100,
        name='On-Time %',
        line=dict(color='green', width=3),
        mode='lines+markers'
    ))
    fig2.add_trace(go.Scatter(
        x=monthly_perf['Year-Month'],
        y=monthly_perf['Late_delivery_risk'] * 100,
        name='Late Risk %',
        line=dict(color='red', width=3),
        mode='lines+markers'
    ))
    fig2.update_layout(
        title='Monthly Performance Trends',
        xaxis_title='Month',
        yaxis_title='Percentage (%)',
        xaxis_tickangle=-45
    )
    fig2.show()
    
    # 3. Delivery Status Pie Chart
    status_counts = df['Delivery Status'].value_counts()
    fig3 = px.pie(
        values=status_counts.values,
        names=status_counts.index,
        title='Orders by Delivery Status',
        color_discrete_sequence=px.colors.qualitative.Set3
    )
    fig3.show()
    
    return fig1, fig2, fig3

def create_risk_visualizations(df):
    """Create risk heatmap visualizations"""
    
    print("\n" + "="*50)
    print("CREATING RISK HEATMAPS")
    print("="*50)
    
    # 1. Market Risk Heatmap
    market_risk_matrix = pd.crosstab(
        df['Market'],
        df['Late_delivery_risk'],
        values=df['Order Id'],
        aggfunc='count',
        normalize='index'
    ) * 100
    
    fig1 = px.imshow(
        market_risk_matrix,
        title='Market Risk Profile (%)',
        labels=dict(x='Risk Level', y='Market', color='Percentage'),
        color_continuous_scale='Reds',
        text_auto='.1f'
    )
    fig1.update_layout(height=400)
    fig1.show()
    
    # 2. Category Risk Heatmap
    category_risk_matrix = pd.crosstab(
        df['Category Name'],
        df['Late_delivery_risk'],
        values=df['Order Id'],
        aggfunc='count',
        normalize='index'
    ) * 100
    
    fig2 = px.imshow(
        category_risk_matrix,
        title='Product Category Risk Profile (%)',
        labels=dict(x='Risk Level', y='Category', color='Percentage'),
        color_continuous_scale='Reds',
        text_auto='.1f'
    )
    fig2.update_layout(height=500)
    fig2.show()
    
    # 3. Multi-dimensional Heatmap (Market × Category × Risk)
    # Create pivot table
    pivot_data = pd.pivot_table(
        df,
        values='Order Id',
        index='Market',
        columns=['Category Name', 'Late_delivery_risk'],
        aggfunc='count',
        fill_value=0
    )
    
    # Prepare data for heatmap
    heatmap_data = []
    for market in df['Market'].unique():
        for category in df['Category Name'].unique():
            for risk in [0, 1]:
                count = len(df[
                    (df['Market'] == market) & 
                    (df['Category Name'] == category) & 
                    (df['Late_delivery_risk'] == risk)
                ])
                if count > 0:
                    heatmap_data.append({
                        'Market': market,
                        'Category': category,
                        'Risk Level': f"{'High Risk' if risk == 1 else 'Low Risk'}",
                        'Order Count': count
                    })
    
    heatmap_df = pd.DataFrame(heatmap_data)
    
    if not heatmap_df.empty:
        fig3 = px.density_heatmap(
            heatmap_df,
            x='Market',
            y='Category',
            z='Order Count',
            facet_col='Risk Level',
            title='Order Volume by Market, Category, and Risk Level',
            color_continuous_scale='Viridis'
        )
        fig3.update_layout(height=500)
        fig3.show()
    
    # 4. Order Status Heatmap
    status_matrix = pd.crosstab(
        df['Market'],
        df['Order Status'],
        values=df['Order Id'],
        aggfunc='count'
    ).fillna(0)
    
    fig4 = px.imshow(
        status_matrix,
        title='Order Status Heatmap by Market',
        labels=dict(x='Order Status', y='Market', color='Order Count'),
        color_continuous_scale='Blues',
        text_auto=True,
        aspect="auto"
    )
    fig4.update_layout(height=400)
    fig4.show()
    
    return fig1, fig2, fig3, fig4

# =============================================================================
# 5. EXPORT SUMMARY REPORTS
# =============================================================================

def export_summary_reports(df, shipping_perf, risk_matrix, filename_prefix='supply_chain_report'):
    """Export summary statistics to CSV"""
    
    print("\n" + "="*50)
    print("EXPORTING SUMMARY REPORTS")
    print("="*50)
    
    # 1. Overall metrics
    overall_metrics = pd.DataFrame({
        'Metric': ['Total Orders', 'On-Time %', 'Avg Shipping Days', 'Late Risk %'],
        'Value': [
            len(df),
            df['On_Time'].mean() * 100,
            df['Days for shipping (real)'].mean(),
            df['Late_delivery_risk'].mean() * 100
        ]
    })
    overall_metrics.to_csv(f'{filename_prefix}_overall_metrics.csv', index=False)
    print(f"✅ Saved: {filename_prefix}_overall_metrics.csv")
    
    # 2. Shipping performance
    shipping_perf.to_csv(f'{filename_prefix}_shipping_performance.csv')
    print(f"✅ Saved: {filename_prefix}_shipping_performance.csv")
    
    # 3. Risk matrix
    risk_matrix.to_csv(f'{filename_prefix}_risk_matrix.csv')
    print(f"✅ Saved: {filename_prefix}_risk_matrix.csv")
    
    # 4. Monthly trends
    monthly_trends = df.groupby('Year-Month').agg({
        'On_Time': 'mean',
        'Late_delivery_risk': 'mean',
        'Order Id': 'count'
    }).round(3)
    monthly_trends.to_csv(f'{filename_prefix}_monthly_trends.csv')
    print(f"✅ Saved: {filename_prefix}_monthly_trends.csv")

# =============================================================================
# 6. MAIN EXECUTION
# =============================================================================

def main():
    """Main function to run the complete analysis"""
    
    print("="*60)
    print("SUPPLY CHAIN PERFORMANCE MONITORING SYSTEM")
    print("Module 1: Logistics Performance + Risk Heat Map")
    print("="*60)
    
    # Load data
    df = load_and_prepare_data()
    
    # Calculate metrics
    shipping_perf, status_df = calculate_logistics_metrics(df)
    market_risk, category_risk, risk_matrix, status_by_market = calculate_risk_metrics(df)
    
    # Create visualizations
    print("\n" + "="*50)
    print("DISPLAYING VISUALIZATIONS")
    print("="*50)
    print("Close each chart window to see the next one...")
    
    logistics_charts = create_logistics_visualizations(df)
    risk_charts = create_risk_visualizations(df)
    
    # Export reports
    export_summary_reports(df, shipping_perf, risk_matrix)
    
    print("\n" + "="*60)
    print("✅ ANALYSIS COMPLETE!")
    print("="*60)
    print("\n📁 Generated files:")
    print("   - supply_chain_report_overall_metrics.csv")
    print("   - supply_chain_report_shipping_performance.csv")
    print("   - supply_chain_report_risk_matrix.csv")
    print("   - supply_chain_report_monthly_trends.csv")
    print("\n📊 Visualizations displayed:")
    print("   - Shipping Mode Performance")
    print("   - Monthly Performance Trends")
    print("   - Delivery Status Distribution")
    print("   - Market Risk Heatmap")
    print("   - Category Risk Heatmap")
    print("   - Multi-dimensional Risk Heatmap")
    print("   - Order Status Heatmap")

# Run the main function
if __name__ == "__main__":
    main()
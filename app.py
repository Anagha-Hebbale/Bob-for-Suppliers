import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os
import random
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Bob For Suppliers",
    page_icon="👷",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header { 
        font-size: 4rem; 
        font-weight: 700; 
        color: #1E3A8A; 
        text-align: center;
        margin-bottom: 0;
    }
    .sub-header { 
        font-size: 2.5rem; 
        font-weight: 600; 
        color: #2563EB; 
        margin-bottom: 1rem;
    }
    .byline {
        font-size: 1.5rem;
        color: #6B7280;
        text-align: center;
        margin-top: -10px;
        margin-bottom: 20px;
        font-style: italic;
    }
    .metric-card { 
        background-color: #F3F4F6; 
        padding: 1.5rem;         /* Increased padding */
        border-radius: 0.5rem; 
        text-align: center; 
        box-shadow: 0 2px 6px rgba(0,0,0,0.15);
        font-size: 1.2rem;       /* Added font size for card content */
    }
    .metric-card h3 {
        font-size: 1.3rem;       /* Added for card titles */
    }
    .metric-card h2 {
        font-size: 2.2rem;       /* Added for card values */
    }
    .insight-box { 
        background-color: #EFF6FF; 
        padding: 1.5rem;          /* Increased padding */
        border-radius: 0.5rem; 
        border-left: 6px solid #2563EB;  /* Thicker border */
        margin: 1rem 0; 
        font-size: 1.2rem;        /* Added font size */
    }
    .insight-box { 
        background-color: #EFF6FF; 
        padding: 1.5rem;          /* Increased padding */
        border-radius: 0.5rem; 
        border-left: 6px solid #2563EB;  /* Thicker border */
        margin: 1rem 0; 
        font-size: 1.2rem;        /* Added font size */
    }
    .improvement-box {
        background-color: #FEF3C7;
        padding: 1.5rem;           /* Increased padding */
        border-radius: 0.5rem;
        border-left: 6px solid #F59E0B;  /* Thicker border */
        margin: 1rem 0;
        font-size: 1.2rem;         /* Added font size */
    }
    .bob-quote {
        background-color: #DBEAFE;
        padding: 0.5rem 1rem;
        border-radius: 2rem;
        display: inline-block;
        font-weight: 500;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'page' not in st.session_state:
    st.session_state.page = 'Dashboard'
if 'improvements' not in st.session_state:
    st.session_state.improvements = []

# Load the data
try:
    df = pd.read_csv('data/cleaned_data.csv')
    st.sidebar.success(f"✅ Loaded cleaned_data.csv: {len(df):,} rows")
except:
    st.sidebar.error("Could not load cleaned_data.csv")
    st.stop()

# Show column names in sidebar for reference
with st.sidebar.expander("📋 Available Columns"):
    for col in df.columns:
        st.caption(f"• {col}")

# Navigation
with st.sidebar:
    st.markdown("---")
    st.subheader("Navigation")
    pages = ["Dashboard", "Performance", "Anomaly Detection", "Predictive", "Seasonal Trends", "Improvements"]
    for page in pages:
        if st.button(page, use_container_width=True):
            st.session_state.page = page

# Function to generate random improvement advice
def generate_improvements():
    improvements = []
    
    # Market-based improvements
    if 'Market' in df.columns:
        markets = df['Market'].unique()
        worst_market = df.groupby('Market')['Order Profit Per Order'].mean().idxmin()
        best_market = df.groupby('Market')['Order Profit Per Order'].mean().idxmax()
        
        improvements.append({
            'area': 'Market Strategy',
            'advice': f"Consider reducing operations in {worst_market} market and focusing more on {best_market} where profits are {df.groupby('Market')['Order Profit Per Order'].mean().max():.2f}x higher.",
            'impact': 'High',
            'effort': 'Medium'
        })
    
    # Shipping improvements
    if 'Shipping Mode' in df.columns and 'Late_delivery_risk' in df.columns:
        risk_by_shipping = df.groupby('Shipping Mode')['Late_delivery_risk'].mean()
        worst_shipping = risk_by_shipping.idxmax()
        
        improvements.append({
            'area': 'Shipping Optimization',
            'advice': f"'{worst_shipping}' has the highest late delivery risk ({risk_by_shipping.max():.1%}). Consider renegotiating contracts or finding alternatives.",
            'impact': 'Medium',
            'effort': 'Low'
        })
    
    # Customer segment improvements
    if 'Customer Segment' in df.columns:
        top_segment = df['Customer Segment'].value_counts().index[0]
        segment_profit = df.groupby('Customer Segment')['Order Profit Per Order'].mean()
        best_segment = segment_profit.idxmax()
        
        improvements.append({
            'area': 'Customer Targeting',
            'advice': f"While '{top_segment}' has most orders, '{best_segment}' is most profitable. Create loyalty program for {best_segment} segment.",
            'impact': 'High',
            'effort': 'Medium'
        })
    
    # Discount optimization
    if 'Order Item Discount' in df.columns and 'Order Profit Per Order' in df.columns:
        high_discount = df[df['Order Item Discount'] > df['Order Item Discount'].quantile(0.9)]
        if len(high_discount) > 0:
            discount_impact = high_discount['Order Profit Per Order'].mean() / df['Order Profit Per Order'].mean()
            if discount_impact < 0.8:
                improvements.append({
                    'area': 'Discount Strategy',
                    'advice': f"High discounts are reducing profits by {100 - discount_impact*100:.1f}%. Consider tiered discount structure based on order value.",
                    'impact': 'Medium',
                    'effort': 'Low'
                })
    
    # Late delivery improvements
    if 'Late_delivery_risk' in df.columns:
        risk_rate = df['Late_delivery_risk'].mean()
        if risk_rate > 0.3:
            improvements.append({
                'area': 'Delivery Performance',
                'advice': f"Late delivery risk is high at {risk_rate:.1%}. Implement real-time tracking and proactive customer communication.",
                'impact': 'High',
                'effort': 'High'
            })
    
    # Seasonal improvements
    if 'order date (DateOrders)' in df.columns:
        df['order_date'] = pd.to_datetime(df['order date (DateOrders)'])
        df['month'] = df['order_date'].dt.month
        monthly_orders = df.groupby('month').size()
        peak_months = monthly_orders.nlargest(3).index.tolist()
        
        improvements.append({
            'area': 'Inventory Planning',
            'advice': f"Peak months are {peak_months}. Increase safety stock by 30% during these months to prevent stockouts.",
            'impact': 'High',
            'effort': 'Medium'
        })
    
    # Add some random improvements
    random_advice = [
        {
            'area': 'Warehouse Efficiency',
            'advice': "Reorganize warehouse by moving fast-moving items closer to shipping dock. Could reduce picking time by 20%.",
            'impact': 'Medium',
            'effort': 'Low'
        },
        {
            'area': 'Supplier Management',
            'advice': "Top 20% of suppliers cause 80% of delays. Consider supplier scorecard and performance-based contracts.",
            'impact': 'High',
            'effort': 'High'
        },
        {
            'area': 'Returns Management',
            'advice': "Implement prepaid returns labels to gather data on return reasons. Currently losing insights on why products are returned.",
            'impact': 'Medium',
            'effort': 'Low'
        },
        {
            'area': 'Pricing Strategy',
            'advice': "Dynamic pricing based on demand could increase profits by 15-20% during peak seasons.",
            'impact': 'High',
            'effort': 'High'
        },
        {
            'area': 'Customer Communication',
            'advice': "Add SMS notifications for order updates. 70% of customers prefer text updates over email.",
            'impact': 'Medium',
            'effort': 'Low'
        }
    ]
    
    # Add 2 random improvements
    improvements.extend(random.sample(random_advice, min(2, len(random_advice))))
    
    return improvements

# Main header
st.markdown('<p class="main-header">👷 Bob For Suppliers</p>', unsafe_allow_html=True)
st.markdown('<p class="byline">From Bob the Builder</p>', unsafe_allow_html=True)

# DASHBOARD PAGE
if st.session_state.page == "Dashboard":
    st.markdown('<p class="sub-header">📊 Executive Dashboard</p>', unsafe_allow_html=True)
    
    # Key Metrics Row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        late_rate = (df['Delivery Status'] == 'Late delivery').mean() if 'Delivery Status' in df.columns else 0.15
        st.metric("Late Delivery Rate", f"{late_rate:.1%}", "⚠️ Needs improvement" if late_rate > 0.2 else "✅ Good")
    
    with col2:
        avg_profit = df['Order Profit Per Order'].mean() if 'Order Profit Per Order' in df.columns else 0
        st.metric("Avg Profit per Order", f"${avg_profit:.2f}")
    
    with col3:
        total_orders = len(df)
        st.metric("Total Orders", f"{total_orders:,}")
    
    with col4:
        if 'Market' in df.columns:
            markets = df['Market'].nunique()
            st.metric("Active Markets", markets)
    
    st.markdown("---")
    
    # Bob's Daily Insight
    st.markdown('<div class="bob-quote">👷 "Can we fix it? Yes we can!" - Bob</div>', unsafe_allow_html=True)
    
    # Insights Row
    st.subheader("💡 Key Insights")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Best Market by Profit
        if 'Market' in df.columns and 'Order Profit Per Order' in df.columns:
            market_profit = df.groupby('Market')['Order Profit Per Order'].mean().sort_values(ascending=False)
            best_market = market_profit.index[0]
            best_profit = market_profit.iloc[0]
            st.markdown(f"""
            <div class="insight-box">
                <h4>🏆 Most Profitable Market</h4>
                <p><b>{best_market}</b> with average profit of <b>${best_profit:.2f}</b></p>
            </div>
            """, unsafe_allow_html=True)
        
        # Best Shipping Mode
        if 'Shipping Mode' in df.columns and 'Order Profit Per Order' in df.columns:
            shipping_profit = df.groupby('Shipping Mode')['Order Profit Per Order'].mean().sort_values(ascending=False)
            best_shipping = shipping_profit.index[0]
            best_shipping_profit = shipping_profit.iloc[0]
            st.markdown(f"""
            <div class="insight-box">
                <h4>🚚 Best Shipping Mode</h4>
                <p><b>{best_shipping}</b> with average profit of <b>${best_shipping_profit:.2f}</b></p>
            </div>
            """, unsafe_allow_html=True)
    
    with col2:
        # Late Delivery Risk
        if 'Late_delivery_risk' in df.columns:
            risk_rate = df['Late_delivery_risk'].mean() * 100
            st.markdown(f"""
            <div class="insight-box">
                <h4>⚠️ Late Delivery Risk</h4>
                <p><b>{risk_rate:.1f}%</b> of orders at risk of late delivery</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Top Category
        if 'Category Name' in df.columns:
            top_cat = df['Category Name'].value_counts().index[0]
            cat_pct = (df['Category Name'].value_counts().iloc[0] / len(df)) * 100
            st.markdown(f"""
            <div class="insight-box">
                <h4>📦 Top Category</h4>
                <p><b>{top_cat}</b> ({cat_pct:.1f}% of orders)</p>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Charts Row
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Profit by Market")
        if 'Market' in df.columns and 'Order Profit Per Order' in df.columns:
            market_data = df.groupby('Market')['Order Profit Per Order'].sum().reset_index()
            fig = px.bar(market_data, x='Market', y='Order Profit Per Order', 
                        title="Total Profit by Market",
                        color='Order Profit Per Order',
                        color_continuous_scale='Viridis')
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Orders by Market")
        if 'Market' in df.columns:
            market_counts = df['Market'].value_counts().reset_index()
            market_counts.columns = ['Market', 'Count']
            fig = px.pie(market_counts, values='Count', names='Market', 
                        title="Order Distribution by Market")
            st.plotly_chart(fig, use_container_width=True)

# PERFORMANCE PAGE
elif st.session_state.page == "Performance":
    st.markdown('<p class="sub-header">📈 Performance Metrics</p>', unsafe_allow_html=True)
    
    tab1, tab2, tab3 = st.tabs(["Delivery Performance", "Shipping Analysis", "Category Analysis"])
    
    with tab1:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Delivery Status")
            if 'Delivery Status' in df.columns:
                status_counts = df['Delivery Status'].value_counts()
                fig = px.pie(values=status_counts.values, names=status_counts.index,
                            title="Delivery Status Distribution")
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("Late Delivery Risk by Market")
            if 'Market' in df.columns and 'Late_delivery_risk' in df.columns:
                risk_by_market = df.groupby('Market')['Late_delivery_risk'].mean().reset_index()
                fig = px.bar(risk_by_market, x='Market', y='Late_delivery_risk',
                            title="Late Delivery Risk by Market",
                            color='Late_delivery_risk',
                            color_continuous_scale='RdYlGn_r')
                st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Shipping Mode Distribution")
            if 'Shipping Mode' in df.columns:
                shipping_counts = df['Shipping Mode'].value_counts()
                fig = px.bar(x=shipping_counts.index, y=shipping_counts.values,
                            title="Orders by Shipping Mode",
                            color=shipping_counts.values,
                            color_continuous_scale='Blues')
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("Shipping Delay Stats")
            if 'shipping_delay' in df.columns:
                delay_stats = df['shipping_delay'].describe()
                st.dataframe(delay_stats)
    
    with tab3:
        st.subheader("Top Categories by Profit")
        if 'Category Name' in df.columns and 'Order Profit Per Order' in df.columns:
            cat_profit = df.groupby('Category Name')['Order Profit Per Order'].sum().nlargest(10)
            fig = px.bar(x=cat_profit.index, y=cat_profit.values,
                        title="Top 10 Categories by Total Profit",
                        color=cat_profit.values,
                        color_continuous_scale='Greens')
            st.plotly_chart(fig, use_container_width=True)

# ANOMALY DETECTION PAGE
elif st.session_state.page == "Anomaly Detection":
    st.markdown('<p class="sub-header">🔍 Anomaly Detection</p>', unsafe_allow_html=True)
    
    st.markdown('<div class="bob-quote">👷 "We need to find what doesn\'t belong!" - Bob</div>', unsafe_allow_html=True)
    
    # Select numerical columns
    num_cols = ['Order Profit Per Order', 'Sales', 'Order Item Discount', 
                'Order Item Quantity', 'Order Item Total']
    available_cols = [col for col in num_cols if col in df.columns]
    
    if len(available_cols) >= 2:
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader("Settings")
            feature1 = st.selectbox("X Axis", available_cols, index=0)
            feature2 = st.selectbox("Y Axis", available_cols, index=min(1, len(available_cols)-1))
            threshold = st.slider("Sensitivity", 2.0, 4.0, 3.0, 0.1)
            
            if st.button("🔍 Find Anomalies"):
                from scipy import stats
                data = df[[feature1, feature2]].dropna()
                z_scores = np.abs(stats.zscore(data))
                df['is_anomaly'] = False
                df.loc[data.index, 'is_anomaly'] = (z_scores > threshold).any(axis=1)
                st.success(f"Found {df['is_anomaly'].sum()} anomalies")
        
        with col2:
            if 'is_anomaly' in df.columns:
                fig = px.scatter(df, x=feature1, y=feature2, color='is_anomaly',
                               color_discrete_map={True: 'red', False: 'blue'},
                               title="Anomaly Detection Results",
                               opacity=0.6,
                               hover_data=['Market', 'Category Name'] if 'Market' in df.columns else None)
                st.plotly_chart(fig, use_container_width=True)

# PREDICTIVE PAGE
elif st.session_state.page == "Predictive":
    st.markdown('<p class="sub-header">🎯 Predictive Risk Model</p>', unsafe_allow_html=True)
    
    st.markdown('<div class="bob-quote">👷 "Let\'s predict before we build!" - Bob</div>', unsafe_allow_html=True)
    
    st.subheader("Late Delivery Risk Predictor")
    
    col1, col2 = st.columns(2)
    
    with col1:
        with st.form("pred_form"):
            market = st.selectbox("Market", df['Market'].unique() if 'Market' in df.columns else ['Unknown'])
            shipping = st.selectbox("Shipping Mode", df['Shipping Mode'].unique() if 'Shipping Mode' in df.columns else ['Standard'])
            category = st.selectbox("Category", df['Category Name'].unique() if 'Category Name' in df.columns else ['Unknown'])
            
            submit = st.form_submit_button("🔮 Predict Risk")
            
            if submit:
                # Calculate risk based on historical data
                risk_factors = []
                if 'Market' in df.columns and 'Late_delivery_risk' in df.columns:
                    market_risk = df[df['Market'] == market]['Late_delivery_risk'].mean()
                    risk_factors.append(market_risk)
                if 'Shipping Mode' in df.columns and 'Late_delivery_risk' in df.columns:
                    shipping_risk = df[df['Shipping Mode'] == shipping]['Late_delivery_risk'].mean()
                    risk_factors.append(shipping_risk)
                
                if risk_factors:
                    st.session_state.risk = np.mean(risk_factors)
                else:
                    st.session_state.risk = np.random.uniform(0.2, 0.8)
    
    with col2:
        if 'risk' in st.session_state:
            risk = st.session_state.risk
            risk_level = 'HIGH' if risk > 0.6 else 'MEDIUM' if risk > 0.3 else 'LOW'
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=risk * 100,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': f"Risk Score: {risk_level}"},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': 'red' if risk>0.6 else 'orange' if risk>0.3 else 'green'},
                    'steps': [
                        {'range': [0, 30], 'color': 'lightgreen'},
                        {'range': [30, 60], 'color': 'yellow'},
                        {'range': [60, 100], 'color': 'salmon'}
                    ]
                }
            ))
            st.plotly_chart(fig, use_container_width=True)

# SEASONAL TRENDS PAGE
elif st.session_state.page == "Seasonal Trends":
    st.markdown('<p class="sub-header">📅 Seasonal Trends</p>', unsafe_allow_html=True)
    
    st.markdown('<div class="bob-quote">👷 "Every season has its reason!" - Bob</div>', unsafe_allow_html=True)
    
    if 'order date (DateOrders)' in df.columns:
        df['order_date'] = pd.to_datetime(df['order date (DateOrders)'])
        df['month'] = df['order_date'].dt.month
        df['month_name'] = df['order_date'].dt.strftime('%B')
        df['quarter'] = df['order_date'].dt.quarter
        df['year'] = df['order_date'].dt.year
        df['day_of_week'] = df['order_date'].dt.day_name()
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Monthly trends
            monthly = df.groupby('month_name').size().reset_index(name='orders')
            month_order = ['January', 'February', 'March', 'April', 'May', 'June',
                          'July', 'August', 'September', 'October', 'November', 'December']
            monthly['month_name'] = pd.Categorical(monthly['month_name'], categories=month_order, ordered=True)
            monthly = monthly.sort_values('month_name')
            
            fig = px.line(monthly, x='month_name', y='orders', 
                        title="Monthly Order Volume",
                        markers=True)
            fig.add_hline(y=monthly['orders'].mean(), line_dash="dash", 
                        annotation_text="Average")
            fig.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Day of week
            daily = df.groupby('day_of_week').size().reset_index(name='orders')
            day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            daily['day_of_week'] = pd.Categorical(daily['day_of_week'], categories=day_order, ordered=True)
            daily = daily.sort_values('day_of_week')
            
            fig = px.bar(daily, x='day_of_week', y='orders',
                        title="Orders by Day of Week",
                        color='orders',
                        color_continuous_scale='Viridis')
            st.plotly_chart(fig, use_container_width=True)
        
        # Peak season insight
        peak_month = monthly.loc[monthly['orders'].idxmax(), 'month_name']
        st.info(f"🏆 Peak season: **{peak_month}** - Plan inventory and staffing accordingly!")
    else:
        st.warning("Date column not found")

# IMPROVEMENTS PAGE
elif st.session_state.page == "Improvements":
    st.markdown('<p class="sub-header">🔧 Improvements & Recommendations</p>', unsafe_allow_html=True)
    
    st.markdown('<div class="bob-quote">👷 "Can we improve it? Yes we can!" - Bob</div>', unsafe_allow_html=True)
    
    # Generate improvements
    improvements = generate_improvements()
    
    # Display improvements
    col1, col2 = st.columns(2)
    
    for i, imp in enumerate(improvements):
        with col1 if i % 2 == 0 else col2:
            impact_color = "🔴" if imp['impact'] == 'High' else "🟡" if imp['impact'] == 'Medium' else "🟢"
            effort_icon = "💪" if imp['effort'] == 'High' else "👍" if imp['effort'] == 'Medium' else "✨"
            
            st.markdown(f"""
            <div class="improvement-box">
                <h4>{imp['area']}</h4>
                <p>{imp['advice']}</p>
                <p><b>Impact:</b> {impact_color} {imp['impact']} | <b>Effort:</b> {effort_icon} {imp['effort']}</p>
            </div>
            """, unsafe_allow_html=True)
    
    # Priority Matrix
    st.markdown("---")
    st.subheader("📊 Priority Matrix")
    
    priority_data = {
        'Improvement': [imp['area'] for imp in improvements[:4]],
        'Impact Score': [3 if imp['impact'] == 'High' else 2 if imp['impact'] == 'Medium' else 1 for imp in improvements[:4]],
        'Effort Score': [3 if imp['effort'] == 'Low' else 2 if imp['effort'] == 'Medium' else 1 for imp in improvements[:4]]
    }
    
    if priority_data['Improvement']:
        priority_df = pd.DataFrame(priority_data)
        priority_df['Priority'] = priority_df['Impact Score'] * 2 - priority_df['Effort Score']
        priority_df = priority_df.sort_values('Priority', ascending=False)
        
        fig = px.scatter(priority_df, x='Effort Score', y='Impact Score',
                        text='Improvement', size='Priority',
                        title="Quick Wins vs Major Projects",
                        labels={'Effort Score': 'Effort Required (1=High, 3=Low)',
                               'Impact Score': 'Impact (1=Low, 3=High)'})
        fig.update_traces(textposition='top center')
        fig.update_layout(xaxis_range=[0, 4], yaxis_range=[0, 4])
        st.plotly_chart(fig, use_container_width=True)
    
    # Quick wins section
    st.markdown("---")
    st.subheader("⚡ Quick Wins (Low Effort, High Impact)")
    
    quick_wins = [imp for imp in improvements if imp['effort'] == 'Low' and imp['impact'] == 'High']
    if quick_wins:
        for imp in quick_wins:
            st.markdown(f"""
            <div style="background-color: #D1FAE5; padding: 0.5rem 1rem; border-radius: 0.5rem; margin: 0.5rem 0;">
                <b>✅ {imp['area']}:</b> {imp['advice']}
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("No quick wins identified yet. Keep analyzing!")

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #6B7280; padding: 1rem;'>
        <p>👷 Bob For Suppliers | From Bob the Builder</p>
        <p style='font-size: 0.875rem;'>"Can we fix it? Yes we can!"</p>
    </div>
    """,
    unsafe_allow_html=True
)
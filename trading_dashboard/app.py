#!/usr/bin/env python3
"""
Interactive Trading Intelligence Dashboard
Visualize and interact with the trading system
"""

import os
import json
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
import sys
sys.path.append('..')  # Add parent directory to path
from trading_intelligence_core import TradingIntelligenceCore, APIKeyManager

class TradingDashboard:
    def __init__(self):
        """Initialize the trading dashboard"""
        self.trading_core = TradingIntelligenceCore()
        self.api_key_manager = APIKeyManager()
        
        # Set up Streamlit page configuration
        st.set_page_config(
            page_title="Trading Intelligence Dashboard",
            page_icon=":chart_with_upwards_trend:",
            layout="wide"
        )
        
        # Add API key management to the dashboard
        self.render_api_key_management()

    def load_recent_results(self):
        """
        Load the most recent trading results
        
        Returns:
            dict: Most recent trading results
        """
        results_dir = self.trading_core.data_dir
        results_files = [
            f for f in os.listdir(results_dir) 
            if f.startswith('trading_results_') and f.endswith('.json')
        ]
        
        if not results_files:
            return None
        
        # Get the most recent results file
        latest_file = max(
            [os.path.join(results_dir, f) for f in results_files], 
            key=os.path.getctime
        )
        
        with open(latest_file, 'r') as f:
            return json.load(f)

    def render_dashboard(self):
        """
        Render the main Streamlit dashboard
        """
        st.title("🚀 Trading Intelligence Dashboard")
        
        # Load recent results
        results = self.load_recent_results()
        
        if not results:
            st.warning("No trading results available. Run the trading intelligence workflow.")
            return

        # Dashboard Sections
        self.performance_overview(results)
        self.model_evaluation_section(results)
        self.strategy_details_section(results)
        self.risk_analysis_section(results)

    def performance_overview(self, results):
        """
        Render performance overview section
        
        Args:
            results (dict): Trading results
        """
        st.header("🔥 Performance Overview")
        
        performance = results['performance_metrics']
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                label="Total Return", 
                value=f"{performance['total_return']:.2%}"
            )
        
        with col2:
            st.metric(
                label="Annual Return", 
                value=f"{performance['annual_return']:.2%}"
            )
        
        with col3:
            st.metric(
                label="Sharpe Ratio", 
                value=f"{performance['sharpe_ratio']:.2f}"
            )
        
        with col4:
            st.metric(
                label="Win Rate", 
                value=f"{performance['win_rate']:.2%}"
            )

    def model_evaluation_section(self, results):
        """
        Render model evaluation section
        
        Args:
            results (dict): Trading results
        """
        st.header("🧠 Model Performance")
        
        model_eval = results['model_evaluation']
        
        # Create a DataFrame for visualization
        model_df = pd.DataFrame.from_dict(
            model_eval, 
            orient='index', 
            columns=['Mean Accuracy', 'Std Accuracy']
        )
        
        # Bar chart of model accuracies
        fig = px.bar(
            model_df, 
            x=model_df.index, 
            y='Mean Accuracy', 
            error_y='Std Accuracy',
            title="Model Accuracy Comparison"
        )
        
        st.plotly_chart(fig, use_container_width=True)

    def strategy_details_section(self, results):
        """
        Render detailed strategy insights
        
        Args:
            results (dict): Trading results
        """
        st.header("📊 Strategy Insights")
        
        # Placeholder for strategy-specific visualizations
        # In a real implementation, you'd have more detailed strategy data
        st.json(results['performance_metrics'])

    def risk_analysis_section(self, results):
        """
        Render risk analysis section
        
        Args:
            results (dict): Trading results
        """
        st.header("🛡️ Risk Analysis")
        
        performance = results['performance_metrics']
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric(
                label="Maximum Drawdown", 
                value=f"{performance.get('max_drawdown', 0):.2%}"
            )
        
        # Potential visualization of risk metrics
        risk_data = [
            {"Metric": "Total Return", "Value": performance['total_return']},
            {"Metric": "Annual Return", "Value": performance['annual_return']},
            {"Metric": "Sharpe Ratio", "Value": performance['sharpe_ratio']},
        ]
        
        risk_df = pd.DataFrame(risk_data)
        
        fig = px.bar(
            risk_df, 
            x='Metric', 
            y='Value', 
            title="Risk and Performance Metrics"
        )
        
        st.plotly_chart(fig, use_container_width=True)

    def render_api_key_management(self):
        """
        Render API key management interface
        """
        st.sidebar.header("🔑 API Key Management")
        
        # Available providers
        providers = self.api_key_manager.supported_providers
        
        # Current key status
        key_status = self.api_key_manager.check_key_availability()
        
        for provider in providers:
            # Create input for each provider
            key_input = st.sidebar.text_input(
                f"Enter {provider.replace('_', ' ').title()} API Key", 
                type="password",
                key=f"{provider}_key_input"
            )
            
            # Add key button
            if st.sidebar.button(f"Add {provider.replace('_', ' ').title()} Key"):
                try:
                    if key_input:
                        self.api_key_manager.add_key(provider, key_input)
                        st.sidebar.success(f"{provider.title()} key added successfully!")
                    else:
                        st.sidebar.error("Please enter a valid API key")
                except ValueError as e:
                    st.sidebar.error(str(e))
            
            # Show key status
            status = "✅ Available" if key_status.get(provider) else "❌ Not Set"
            st.sidebar.text(f"{provider.replace('_', ' ').title()} Key: {status}")
        
        # Free API Key Resources
        st.sidebar.markdown("### 🌐 Free API Key Resources")
        free_resources = {
            "Finnhub": "https://finnhub.io/",
            "Alpha Vantage": "https://www.alphavantage.co/",
            "Polygon": "https://polygon.io/",
        }
        
        for name, link in free_resources.items():
            st.sidebar.markdown(f"- [{name}]({link})")

    def run(self):
        """
        Run the Streamlit dashboard
        """
        self.render_dashboard()

def main():
    dashboard = TradingDashboard()
    dashboard.run()

if __name__ == '__main__':
    main()
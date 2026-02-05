# ui/dashboard.py
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
from datetime import datetime, timedelta
import numpy as np

class DashboardUI:
    """Dashboard visualization components"""
    
    def __init__(self):
        self.color_scheme = px.colors.sequential.Viridis
        
    def show_live_activity(self):
        """Show live crawling activity"""
        # Simulate live data (in real app, this would come from database)
        time_points = pd.date_range(start='2024-01-01', periods=24, freq='H')
        pages_per_hour = np.random.poisson(50, 24)
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=time_points,
            y=pages_per_hour,
            mode='lines+markers',
            name='Pages/Hour',
            line=dict(color=self.color_scheme[0], width=3),
            marker=dict(size=8)
        ))
        
        fig.update_layout(
            title="Crawling Activity (Last 24 Hours)",
            xaxis_title="Time",
            yaxis_title="Pages Crawled",
            hovermode='x unified',
            template='plotly_dark' if st.get_option('theme.base') == 'dark' else 'plotly_white'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
    def show_crawl_progress(self):
        """Show crawl progress metrics"""
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Progress bar for queue
            queue_size = len(st.session_state.get('url_queue', []))
            max_queue = max(queue_size, 1000)
            st.progress(min(queue_size / max_queue, 1.0), text=f"Queue: {queue_size}")
            
        with col2:
            # Success rate
            if st.session_state.get('processed_urls', 0) > 0:
                success_rate = 0.85  # This would come from actual data
                st.metric("Success Rate", f"{success_rate:.1%}")
            else:
                st.metric("Success Rate", "0%")
                
        with col3:
            # Avg relevance
            avg_relevance = 0.65  # This would come from actual data
            st.metric("Avg Relevance", f"{avg_relevance:.1%}")
            
        # Domain distribution pie chart
        domains = {'example.com': 45, 'test.com': 25, 'demo.org': 15, 'other': 15}
        
        fig = go.Figure(data=[go.Pie(
            labels=list(domains.keys()),
            values=list(domains.values()),
            hole=.3,
            marker_colors=self.color_scheme
        )])
        
        fig.update_layout(
            title="Top Domains Distribution",
            showlegend=True
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
    def show_domain_distribution(self):
        """Show domain distribution visualization"""
        # Sample data
        domains = ['example.com', 'test.com', 'demo.org', 'sample.net', 'website.io']
        pages = [120, 85, 60, 45, 30]
        relevance = [0.8, 0.7, 0.6, 0.9, 0.5]
        
        df = pd.DataFrame({
            'Domain': domains,
            'Pages': pages,
            'Avg Relevance': relevance
        })
        
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Pages per Domain', 'Relevance per Domain'),
            specs=[[{'type': 'bar'}, {'type': 'bar'}]]
        )
        
        fig.add_trace(
            go.Bar(x=domains, y=pages, name='Pages', marker_color=self.color_scheme[0]),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Bar(x=domains, y=relevance, name='Relevance', marker_color=self.color_scheme[4]),
            row=1, col=2
        )
        
        fig.update_layout(
            title="Domain Analysis",
            showlegend=False,
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
    def show_error_tracking(self):
        """Show error tracking visualization"""
        error_types = ['Timeout', '404', '403', '429', '500', 'Network']
        error_counts = [12, 8, 5, 3, 2, 4]
        
        fig = go.Figure(data=[go.Bar(
            x=error_types,
            y=error_counts,
            marker_color=self.color_scheme,
            text=error_counts,
            textposition='auto'
        )])
        
        fig.update_layout(
            title="Error Distribution",
            xaxis_title="Error Type",
            yaxis_title="Count",
            showlegend=False
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Error timeline
        st.subheader("Error Timeline")
        error_data = pd.DataFrame({
            'timestamp': pd.date_range(start='2024-01-01', periods=48, freq='H'),
            'errors': np.random.poisson(2, 48)
        })
        
        fig2 = px.line(
            error_data, 
            x='timestamp', 
            y='errors',
            title="Errors Over Time"
        )
        
        st.plotly_chart(fig2, use_container_width=True)
        
    def show_recent_discoveries(self):
        """Show recently discovered pages"""
        # Sample data
        recent_data = [
            {'url': 'https://example.com/page1', 'title': 'Example Page 1', 'relevance': 0.9, 'time': '10:30'},
            {'url': 'https://test.com/article', 'title': 'Test Article', 'relevance': 0.8, 'time': '10:25'},
            {'url': 'https://demo.org/blog', 'title': 'Demo Blog', 'relevance': 0.7, 'time': '10:20'},
            {'url': 'https://sample.net/tutorial', 'title': 'Sample Tutorial', 'relevance': 0.6, 'time': '10:15'},
            {'url': 'https://website.io/guide', 'title': 'Website Guide', 'relevance': 0.5, 'time': '10:10'},
        ]
        
        for item in recent_data:
            with st.container():
                col1, col2, col3 = st.columns([3, 1, 1])
                with col1:
                    st.write(f"**{item['title']}**")
                    st.caption(item['url'])
                with col2:
                    st.metric("Relevance", f"{item['relevance']:.0%}")
                with col3:
                    st.caption(item['time'])
                st.divider()
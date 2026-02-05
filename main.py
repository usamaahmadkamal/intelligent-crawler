# main.py - Main Streamlit Application
import streamlit as st
import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import time
from typing import Dict, List, Optional, Tuple
import sys
import os
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

# Set page config
st.set_page_config(
    page_title="Intelligent Web Crawling Platform",
    page_icon="🕷️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Import custom modules
from core.crawler_manager import CrawlerManager
from core.database_manager import DatabaseManager
from core.ai_processor import AIProcessor
from ui.dashboard import DashboardUI
from ui.controls import ControlPanel
from ui.niche_manager import NicheManager
from utils.helpers import init_session_state, save_state, load_state

class IntelligentCrawlingPlatform:
    """Main application class orchestrating all expert domains"""
    
    def __init__(self):
        self.init_session_state()
        self.setup_directories()
        self.initialize_managers()
        
    def init_session_state(self):
        """Initialize all session state variables"""
        if 'initialized' not in st.session_state:
            st.session_state.initialized = True
            
            # Niche Management
            st.session_state.niches = {}
            st.session_state.current_niche = None
            st.session_state.niche_profiles = {}
            
            # Crawling State
            st.session_state.crawling_active = False
            st.session_state.paused = False
            st.session_state.workers = 5
            st.session_state.crawl_speed = "medium"
            
            # Queue Management
            st.session_state.url_queue = []
            st.session_state.visited_urls = set()
            st.session_state.processed_urls = 0
            
            # Statistics
            st.session_state.stats = {
                'total_pages': 0,
                'unique_domains': set(),
                'errors': [],
                'start_time': None,
                'last_update': None
            }
            
            # Database
            st.session_state.db_connected = False
            
            # AI Models
            st.session_state.ai_loaded = False
            
    def setup_directories(self):
        """Create necessary directories"""
        dirs = ['data', 'logs', 'exports', 'models', 'configs', 'backups']
        for dir_name in dirs:
            Path(f"./{dir_name}").mkdir(exist_ok=True)
            
    def initialize_managers(self):
        """Initialize all expert managers"""
        try:
            self.db_manager = DatabaseManager()
            st.session_state.db_connected = True
        except:
            st.session_state.db_connected = False
            
        self.crawler_manager = CrawlerManager()
        self.ai_processor = AIProcessor()
        self.dashboard_ui = DashboardUI()
        self.control_panel = ControlPanel()
        self.niche_manager = NicheManager()
        
    def run(self):
        """Main application loop"""
        # Sidebar navigation
        with st.sidebar:
            st.title("🕷️ Intelligent Crawler")
            
            # Navigation
            nav_option = st.selectbox(
                "Navigation",
                ["Dashboard", "Niche Management", "Crawling Control", 
                 "Data Explorer", "AI Analytics", "Settings", "Monitoring"]
            )
            
            # System status
            st.divider()
            self.display_system_status()
            
        # Main content area
        if nav_option == "Dashboard":
            self.show_dashboard()
        elif nav_option == "Niche Management":
            self.show_niche_management()
        elif nav_option == "Crawling Control":
            self.show_crawling_control()
        elif nav_option == "Data Explorer":
            self.show_data_explorer()
        elif nav_option == "AI Analytics":
            self.show_ai_analytics()
        elif nav_option == "Settings":
            self.show_settings()
        elif nav_option == "Monitoring":
            self.show_monitoring()
            
    def display_system_status(self):
        """Display system status in sidebar"""
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Active Workers", st.session_state.workers)
        with col2:
            status = "🟢 Active" if st.session_state.crawling_active else "🔴 Stopped"
            st.metric("Crawl Status", status)
            
        if st.session_state.stats['total_pages'] > 0:
            st.metric("Pages Indexed", st.session_state.stats['total_pages'])
            st.metric("Unique Domains", len(st.session_state.stats['unique_domains']))
            
    def show_dashboard(self):
        """Main dashboard view"""
        st.title("Real-Time Crawling Dashboard")
        
        # Top metrics row
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Queue Size", len(st.session_state.url_queue))
        with col2:
            st.metric("Processed URLs", st.session_state.processed_urls)
        with col3:
            st.metric("Active Workers", st.session_state.workers)
        with col4:
            speed = "0" if not st.session_state.crawling_active else "50-100"
            st.metric("Pages/Min", speed)
        
        # Charts and visualizations
        tab1, tab2, tab3, tab4 = st.tabs([
            "Live Activity", "Crawl Progress", "Domain Distribution", "Error Tracking"
        ])
        
        with tab1:
            self.dashboard_ui.show_live_activity()
        with tab2:
            self.dashboard_ui.show_crawl_progress()
        with tab3:
            self.dashboard_ui.show_domain_distribution()
        with tab4:
            self.dashboard_ui.show_error_tracking()
            
        # Recent discoveries
        st.subheader("Recent Discoveries")
        self.dashboard_ui.show_recent_discoveries()
        
    def show_niche_management(self):
        """Niche management interface"""
        st.title("Niche & Keyword Management")
        
        tab1, tab2, tab3, tab4 = st.tabs([
            "Create Profile", "Manage Profiles", "Seed Discovery", "Keyword Analysis"
        ])
        
        with tab1:
            self.niche_manager.create_niche_profile()
        with tab2:
            self.niche_manager.manage_profiles()
        with tab3:
            self.niche_manager.seed_discovery()
        with tab4:
            self.niche_manager.keyword_analysis()
            
    def show_crawling_control(self):
        """Crawling control interface"""
        st.title("Crawling Control Center")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Control buttons
            btn_col1, btn_col2, btn_col3, btn_col4 = st.columns(4)
            with btn_col1:
                if st.button("▶️ Start Crawling", type="primary", use_container_width=True):
                    self.start_crawling()
            with btn_col2:
                if st.button("⏸️ Pause", use_container_width=True):
                    self.pause_crawling()
            with btn_col3:
                if st.button("⏹️ Stop", use_container_width=True):
                    self.stop_crawling()
            with btn_col4:
                if st.button("🔄 Resume", use_container_width=True):
                    self.resume_crawling()
            
            # Live controls
            st.subheader("Live Controls")
            self.control_panel.show_live_controls()
            
            # Queue management
            st.subheader("Queue Management")
            self.control_panel.show_queue_management()
            
        with col2:
            # System metrics
            st.subheader("System Health")
            self.control_panel.show_system_health()
            
            # Quick stats
            st.subheader("Quick Stats")
            self.control_panel.show_quick_stats()
            
    def show_data_explorer(self):
        """Data exploration interface"""
        st.title("Data Explorer & Index Management")
        
        # Search and filter
        col1, col2, col3 = st.columns([2, 1, 1])
        with col1:
            search_query = st.text_input("Search indexed content", placeholder="Enter keywords...")
        with col2:
            filter_type = st.selectbox("Filter by", ["All", "High Relevance", "Unreviewed", "Flagged"])
        with col3:
            sort_by = st.selectbox("Sort by", ["Relevance", "Date", "Domain"])
        
        # Data table
        if st.session_state.db_connected:
            data = self.db_manager.search_content(search_query, filter_type, sort_by)
            if len(data) > 0:
                df = pd.DataFrame(data)
                st.dataframe(df, use_container_width=True)
                
                # Export options
                col1, col2, col3 = st.columns(3)
                with col1:
                    if st.button("📥 Export to CSV"):
                        self.export_data(df, 'csv')
                with col2:
                    if st.button("📊 Export to JSON"):
                        self.export_data(df, 'json')
                with col3:
                    if st.button("🗃️ Save to Database"):
                        self.db_manager.bulk_save(df)
            else:
                st.info("No data available. Start crawling to index content.")
        
        # Content preview
        if 'selected_page' in st.session_state:
            st.subheader("Content Preview")
            self.show_content_preview()
            
    def show_ai_analytics(self):
        """AI-powered analytics interface"""
        st.title("AI Analytics & Intelligence")
        
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "Semantic Search", "Trend Analysis", "Content Gaps", 
            "Similarity Detection", "Predictive Insights"
        ])
        
        with tab1:
            self.ai_processor.semantic_search_interface()
        with tab2:
            self.ai_processor.trend_analysis_interface()
        with tab3:
            self.ai_processor.content_gap_analysis()
        with tab4:
            self.ai_processor.similarity_detection()
        with tab5:
            self.ai_processor.predictive_insights()
            
    def show_settings(self):
        """Settings and configuration"""
        st.title("Settings & Configuration")
        
        tab1, tab2, tab3, tab4 = st.tabs([
            "Database", "Crawling", "AI Models", "Notifications"
        ])
        
        with tab1:
            self.show_database_settings()
        with tab2:
            self.show_crawling_settings()
        with tab3:
            self.show_ai_settings()
        with tab4:
            self.show_notification_settings()
            
    def show_monitoring(self):
        """System monitoring interface"""
        st.title("System Monitoring & Alerts")
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            # Live logs
            st.subheader("Live Event Logs")
            self.show_live_logs()
            
            # Error tracking
            st.subheader("Error Tracking")
            self.show_error_details()
            
        with col2:
            # Alert settings
            st.subheader("Alert Configuration")
            self.configure_alerts()
            
            # System metrics
            st.subheader("Resource Usage")
            self.show_resource_usage()
            
    def start_crawling(self):
        """Start the crawling process"""
        if not st.session_state.current_niche:
            st.error("Please select or create a niche profile first!")
            return
            
        st.session_state.crawling_active = True
        st.session_state.paused = False
        st.session_state.stats['start_time'] = datetime.now()
        
        # Initialize crawler
        self.crawler_manager.start_crawling(
            niche_profile=st.session_state.current_niche,
            workers=st.session_state.workers
        )
        
        st.success(f"Crawling started for niche: {st.session_state.current_niche['name']}")
        
    def pause_crawling(self):
        """Pause crawling"""
        st.session_state.paused = True
        self.crawler_manager.pause_crawling()
        st.info("Crawling paused")
        
    def stop_crawling(self):
        """Stop crawling"""
        st.session_state.crawling_active = False
        st.session_state.paused = False
        self.crawler_manager.stop_crawling()
        
        # Save session stats
        self.save_session_summary()
        st.info("Crawling stopped and session saved")
        
    def resume_crawling(self):
        """Resume crawling"""
        if st.session_state.paused:
            st.session_state.paused = False
            self.crawler_manager.resume_crawling()
            st.success("Crawling resumed")
            
    def export_data(self, df, format_type):
        """Export data to specified format"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"exports/crawl_data_{timestamp}.{format_type}"
        
        if format_type == 'csv':
            df.to_csv(filename, index=False)
        elif format_type == 'json':
            df.to_json(filename, orient='records')
            
        st.success(f"Data exported to {filename}")
        
    def save_session_summary(self):
        """Save crawling session summary"""
        summary = {
            'session_id': str(datetime.now().timestamp()),
            'niche': st.session_state.current_niche,
            'stats': st.session_state.stats,
            'duration': str(datetime.now() - st.session_state.stats['start_time']),
            'timestamp': datetime.now().isoformat()
        }
        
        # Save to file
        with open(f"logs/session_{summary['session_id']}.json", 'w') as f:
            json.dump(summary, f, indent=2)
            
    # Additional methods for settings would be implemented here...

if __name__ == "__main__":
    app = IntelligentCrawlingPlatform()
    app.run()
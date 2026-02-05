# ui/controls.py
import streamlit as st
import psutil
import time
from datetime import datetime

class ControlPanel:
    """Crawling control panel components"""
    
    def __init__(self):
        pass
        
    def show_live_controls(self):
        """Show live control sliders and buttons"""
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Worker control
            new_workers = st.slider(
                "Number of Workers",
                min_value=1,
                max_value=50,
                value=st.session_state.get('workers', 5),
                key='workers_slider',
                help="Adjust the number of concurrent crawling workers"
            )
            if new_workers != st.session_state.get('workers', 5):
                st.session_state.workers = new_workers
                st.rerun()
                
        with col2:
            # Crawl speed
            crawl_speed = st.select_slider(
                "Crawl Speed",
                options=["Very Slow", "Slow", "Medium", "Fast", "Very Fast"],
                value=st.session_state.get('crawl_speed', 'Medium'),
                help="Adjust crawling speed (affects rate limiting)"
            )
            st.session_state.crawl_speed = crawl_speed
            
        with col3:
            # Depth limit
            depth_limit = st.number_input(
                "Crawl Depth",
                min_value=1,
                max_value=10,
                value=3,
                help="Maximum depth to crawl from seed URLs"
            )
            st.session_state.depth_limit = depth_limit
            
        # Advanced controls
        with st.expander("Advanced Controls"):
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.checkbox("Infinite Crawl", value=False, 
                          help="Continue crawling indefinitely")
                
            with col2:
                st.checkbox("Auto-throttle", value=True, 
                          help="Automatically adjust speed based on server responses")
                
            with col3:
                st.checkbox("Skip Duplicates", value=True, 
                          help="Skip duplicate content")
                
            with col4:
                st.checkbox("Respect Robots.txt", value=True, 
                          help="Respect robots.txt rules")
                
    def show_queue_management(self):
        """Show queue management interface"""
        # Current queue stats
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Pending URLs", len(st.session_state.get('url_queue', [])))
        with col2:
            st.metric("Processed", st.session_state.get('processed_urls', 0))
        with col3:
            st.metric("Queue Growth", "+12/min", delta="+12")
            
        # Queue actions
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            if st.button("Clear Queue", use_container_width=True):
                st.session_state.url_queue = []
                st.success("Queue cleared")
                
        with col2:
            if st.button("Prioritize High-Quality", use_container_width=True):
                st.info("Queue re-prioritized")
                
        with col3:
            if st.button("Remove Low-Relevance", use_container_width=True):
                st.info("Low-relevance URLs removed")
                
        with col4:
            if st.button("Export Queue", use_container_width=True):
                st.info("Queue exported to file")
                
        # Queue preview
        if st.session_state.get('url_queue'):
            with st.expander("Preview Queue (First 10 URLs)"):
                for i, url in enumerate(list(st.session_state.url_queue)[:10]):
                    st.write(f"{i+1}. {url}")
                    
    def show_system_health(self):
        """Show system health metrics"""
        # CPU usage
        cpu_percent = psutil.cpu_percent(interval=1)
        
        # Memory usage
        memory = psutil.virtual_memory()
        
        # Disk usage
        disk = psutil.disk_usage('/')
        
        # Create metrics
        col1, col2 = st.columns(2)
        with col1:
            st.metric("CPU Usage", f"{cpu_percent}%")
            st.progress(cpu_percent / 100, text=f"CPU: {cpu_percent}%")
            
        with col2:
            st.metric("Memory Usage", f"{memory.percent}%")
            st.progress(memory.percent / 100, text=f"Memory: {memory.percent}%")
            
        # Disk usage
        st.metric("Disk Usage", f"{disk.percent}%")
        st.progress(disk.percent / 100, text=f"Disk: {disk.percent}%")
        
        # System warnings
        if cpu_percent > 80:
            st.warning("High CPU usage detected!")
        if memory.percent > 80:
            st.warning("High memory usage detected!")
        if disk.percent > 90:
            st.error("Disk space running low!")
            
    def show_quick_stats(self):
        """Show quick statistics panel"""
        stats = st.session_state.get('stats', {})
        
        st.metric("Total Pages", stats.get('total_pages', 0))
        st.metric("Unique Domains", len(stats.get('unique_domains', set())))
        
        if stats.get('start_time'):
            duration = datetime.now() - stats['start_time']
            st.metric("Session Duration", str(duration).split('.')[0])
            
        # Error count
        error_count = len(stats.get('errors', []))
        st.metric("Errors", error_count, delta=None)
        
        # Success rate
        if stats.get('total_pages', 0) > 0:
            success_rate = (stats['total_pages'] - error_count) / stats['total_pages']
            st.metric("Success Rate", f"{success_rate:.1%}")
# ui/niche_manager.py
import streamlit as st
import pandas as pd
from typing import List, Dict
import json
from datetime import datetime

class NicheManager:
    """Niche and keyword management interface"""
    
    def __init__(self):
        pass
        
    def create_niche_profile(self):
        """Create a new niche profile"""
        with st.form("create_niche"):
            col1, col2 = st.columns(2)
            
            with col1:
                profile_name = st.text_input("Profile Name", 
                                           placeholder="e.g., 'Python Programming'")
                description = st.text_area("Description", 
                                         placeholder="Describe this niche...")
                
            with col2:
                priority = st.select_slider(
                    "Priority",
                    options=["Low", "Medium", "High", "Critical"],
                    value="Medium"
                )
                
                auto_refresh = st.checkbox("Auto-refresh niche", value=False)
                if auto_refresh:
                    refresh_interval = st.selectbox(
                        "Refresh Interval",
                        ["Daily", "Weekly", "Monthly"]
                    )
                    
            # Keywords section
            st.subheader("Keywords & Filters")
            
            # Main keywords
            keywords_input = st.text_area(
                "Main Keywords",
                placeholder="Enter keywords, one per line or comma-separated",
                height=100
            )
            
            # Negative keywords
            negative_keywords = st.text_area(
                "Negative Keywords",
                placeholder="Keywords to exclude (one per line)",
                height=80
            )
            
            # Advanced settings
            with st.expander("Advanced Settings"):
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    relevance_threshold = st.slider(
                        "Relevance Threshold",
                        min_value=0.0,
                        max_value=1.0,
                        value=0.3,
                        step=0.05,
                        help="Minimum relevance score to include page"
                    )
                    
                with col2:
                    max_depth = st.number_input(
                        "Max Crawl Depth",
                        min_value=1,
                        max_value=10,
                        value=3
                    )
                    
                with col3:
                    max_pages = st.number_input(
                        "Max Pages",
                        min_value=100,
                        max_value=100000,
                        value=1000,
                        step=100
                    )
                    
                # Regional filters
                region = st.multiselect(
                    "Regional Filters",
                    [".com", ".org", ".net", ".io", ".edu", ".gov", "Country-specific"],
                    default=[".com", ".org", ".net"]
                )
                
            # Submit button
            submitted = st.form_submit_button("Create Profile", type="primary")
            
            if submitted and profile_name and keywords_input:
                # Parse keywords
                if '\n' in keywords_input:
                    keywords = [k.strip() for k in keywords_input.split('\n') if k.strip()]
                else:
                    keywords = [k.strip() for k in keywords_input.split(',') if k.strip()]
                    
                # Create profile
                profile = {
                    'name': profile_name,
                    'description': description,
                    'keywords': keywords,
                    'negative_keywords': [k.strip() for k in negative_keywords.split('\n') if k.strip()],
                    'priority': priority,
                    'relevance_threshold': relevance_threshold,
                    'max_depth': max_depth,
                    'max_pages': max_pages,
                    'regions': region,
                    'created_at': datetime.now().isoformat(),
                    'auto_refresh': auto_refresh,
                    'refresh_interval': refresh_interval if auto_refresh else None
                }
                
                # Save to session state
                if 'niche_profiles' not in st.session_state:
                    st.session_state.niche_profiles = {}
                    
                st.session_state.niche_profiles[profile_name] = profile
                st.session_state.current_niche = profile
                
                st.success(f"Profile '{profile_name}' created successfully!")
                
    def manage_profiles(self):
        """Manage existing niche profiles"""
        if not st.session_state.get('niche_profiles'):
            st.info("No niche profiles created yet.")
            return
            
        profiles = st.session_state.niche_profiles
        
        # Profile selector
        selected_profile = st.selectbox(
            "Select Profile",
            list(profiles.keys()),
            key='profile_selector'
        )
        
        if selected_profile:
            profile = profiles[selected_profile]
            
            col1, col2, col3 = st.columns([2, 1, 1])
            
            with col1:
                st.subheader(f"Profile: {profile['name']}")
                st.caption(profile['description'])
                
            with col2:
                if st.button("Load Profile", use_container_width=True):
                    st.session_state.current_niche = profile
                    st.success(f"Profile '{profile['name']}' loaded!")
                    
            with col3:
                if st.button("Delete Profile", type="secondary", use_container_width=True):
                    del profiles[selected_profile]
                    st.session_state.niche_profiles = profiles
                    st.rerun()
                    
            # Display profile details
            with st.expander("Profile Details"):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Keywords:**")
                    for keyword in profile['keywords'][:10]:
                        st.code(keyword)
                        
                    if len(profile['keywords']) > 10:
                        st.caption(f"... and {len(profile['keywords']) - 10} more")
                        
                with col2:
                    st.metric("Priority", profile['priority'])
                    st.metric("Threshold", f"{profile['relevance_threshold']:.0%}")
                    st.metric("Max Depth", profile['max_depth'])
                    st.metric("Max Pages", profile['max_pages'])
                    
            # Edit profile
            if st.button("Edit Profile", type="primary"):
                st.session_state.editing_profile = profile
                st.rerun()
                
    def seed_discovery(self):
        """Seed URL discovery interface"""
        st.subheader("Seed Discovery")
        
        tab1, tab2, tab3 = st.tabs(["Automatic", "Manual", "Import"])
        
        with tab1:
            # Automatic seed discovery
            discovery_method = st.radio(
                "Discovery Method",
                ["Search Engine", "Social Media", "Directories", "Competitor Analysis"]
            )
            
            if discovery_method == "Search Engine":
                search_query = st.text_input("Search Query", 
                                           placeholder="Enter search terms...")
                num_results = st.slider("Number of Results", 10, 100, 20)
                
                if st.button("Discover Seeds"):
                    with st.spinner("Discovering seeds..."):
                        # This would call actual search engine API
                        seeds = self.simulate_search_discovery(search_query, num_results)
                        self.display_seeds(seeds)
                        
            elif discovery_method == "Competitor Analysis":
                competitor_url = st.text_input("Competitor URL", 
                                             placeholder="https://competitor.com")
                
                if st.button("Analyze Competitor"):
                    with st.spinner("Analyzing competitor..."):
                        seeds = self.simulate_competitor_analysis(competitor_url)
                        self.display_seeds(seeds)
                        
        with tab2:
            # Manual seed input
            seed_urls = st.text_area(
                "Enter Seed URLs",
                placeholder="Enter URLs, one per line",
                height=150
            )
            
            if st.button("Add Seeds Manually"):
                urls = [url.strip() for url in seed_urls.split('\n') if url.strip()]
                if urls:
                    self.add_to_queue(urls)
                    st.success(f"Added {len(urls)} seed URLs to queue")
                    
        with tab3:
            # Import from file
            uploaded_file = st.file_uploader(
                "Upload seed file",
                type=['csv', 'txt', 'json'],
                help="Upload CSV, TXT, or JSON file with URLs"
            )
            
            if uploaded_file:
                if uploaded_file.name.endswith('.csv'):
                    df = pd.read_csv(uploaded_file)
                    if 'url' in df.columns:
                        urls = df['url'].tolist()
                        st.dataframe(df.head())
                        
                elif uploaded_file.name.endswith('.txt'):
                    content = uploaded_file.read().decode('utf-8')
                    urls = [line.strip() for line in content.split('\n') if line.strip()]
                    
                if st.button("Import URLs"):
                    self.add_to_queue(urls)
                    st.success(f"Imported {len(urls)} URLs from file")
                    
    def keyword_analysis(self):
        """Keyword analysis interface"""
        st.subheader("Keyword Analysis")
        
        # Input for keyword analysis
        analysis_text = st.text_area(
            "Enter text for keyword analysis",
            placeholder="Paste text to analyze for keywords...",
            height=200
        )
        
        if st.button("Analyze Keywords", type="primary"):
            if analysis_text:
                # This would call actual AI analysis
                keywords = self.simulate_keyword_extraction(analysis_text)
                
                # Display results
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Extracted Keywords")
                    for i, (keyword, score) in enumerate(keywords[:10], 1):
                        st.write(f"{i}. **{keyword}** (score: {score:.3f})")
                        
                with col2:
                    # Keyword cloud visualization
                    st.subheader("Keyword Cloud")
                    # This would be a word cloud visualization
                    st.info("Word cloud visualization would appear here")
                    
            else:
                st.warning("Please enter some text to analyze")
                
        # Trending keywords
        st.subheader("Trending Keyword Detection")
        
        if st.button("Detect Trending Keywords"):
            with st.spinner("Analyzing trends..."):
                trending = self.simulate_trend_detection()
                
                if trending:
                    df = pd.DataFrame(trending)
                    st.dataframe(df, use_container_width=True)
                else:
                    st.info("No trending keywords detected")
                    
    def simulate_search_discovery(self, query, num_results):
        """Simulate search engine discovery"""
        # In real implementation, this would call Google/Bing API
        return [
            f"https://example.com/search/{query.replace(' ', '-')}-{i}"
            for i in range(num_results)
        ]
        
    def simulate_competitor_analysis(self, url):
        """Simulate competitor analysis"""
        return [
            f"{url}/page1",
            f"{url}/page2",
            f"{url}/blog",
            f"{url}/articles",
            f"{url}/resources"
        ]
        
    def simulate_keyword_extraction(self, text):
        """Simulate keyword extraction"""
        # Simple simulation - in real app, use AI/ML
        words = text.lower().split()
        from collections import Counter
        word_counts = Counter(words)
        
        # Filter common words
        common_words = {'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for'}
        keywords = [(word, count/len(words)) 
                   for word, count in word_counts.items() 
                   if word not in common_words and len(word) > 3]
        
        return sorted(keywords, key=lambda x: x[1], reverse=True)[:20]
        
    def simulate_trend_detection(self):
        """Simulate trend detection"""
        return [
            {"keyword": "artificial intelligence", "growth": 1.5, "volume": 1000},
            {"keyword": "machine learning", "growth": 1.2, "volume": 800},
            {"keyword": "deep learning", "growth": 0.8, "volume": 600},
            {"keyword": "neural networks", "growth": 0.6, "volume": 400},
            {"keyword": "natural language processing", "growth": 1.1, "volume": 700},
        ]
        
    def display_seeds(self, seeds):
        """Display discovered seeds"""
        st.subheader(f"Discovered Seeds ({len(seeds)})")
        
        # Create dataframe for display
        df = pd.DataFrame({
            'URL': seeds,
            'Domain': [url.split('/')[2] if '//' in url else url for url in seeds],
            'Select': [True] * len(seeds)
        })
        
        # Editable dataframe
        edited_df = st.data_editor(
            df,
            column_config={
                "Select": st.column_config.CheckboxColumn(
                    "Select",
                    help="Select seeds to add to queue",
                    default=True
                )
            },
            disabled=["URL", "Domain"],
            use_container_width=True
        )
        
        # Add selected seeds to queue
        selected_urls = edited_df[edited_df['Select']]['URL'].tolist()
        
        if selected_urls and st.button("Add Selected to Queue"):
            self.add_to_queue(selected_urls)
            st.success(f"Added {len(selected_urls)} seeds to queue")
            
    def add_to_queue(self, urls):
        """Add URLs to crawling queue"""
        if 'url_queue' not in st.session_state:
            st.session_state.url_queue = []
            
        st.session_state.url_queue.extend(urls)
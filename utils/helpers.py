# utils/helpers.py
import streamlit as st
import json
import pickle
from datetime import datetime, timedelta
from typing import Dict, List, Any
import hashlib
import os
from pathlib import Path

def init_session_state():
    """Initialize session state if not already done"""
    defaults = {
        'initialized': False,
        'niches': {},
        'current_niche': None,
        'crawling_active': False,
        'paused': False,
        'workers': 5,
        'url_queue': [],
        'visited_urls': set(),
        'processed_urls': 0,
        'stats': {
            'total_pages': 0,
            'unique_domains': set(),
            'errors': [],
            'start_time': None,
            'last_update': None
        }
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value
            
def save_state(filename: str = "session_state.pkl"):
    """Save session state to file"""
    try:
        # Convert sets to lists for serialization
        state_copy = dict(st.session_state)
        
        if 'visited_urls' in state_copy:
            state_copy['visited_urls'] = list(state_copy['visited_urls'])
            
        if 'unique_domains' in state_copy.get('stats', {}):
            state_copy['stats']['unique_domains'] = list(
                state_copy['stats']['unique_domains']
            )
            
        with open(f"backups/{filename}", 'wb') as f:
            pickle.dump(state_copy, f)
            
        return True
    except Exception as e:
        print(f"Error saving state: {e}")
        return False
        
def load_state(filename: str = "session_state.pkl"):
    """Load session state from file"""
    try:
        path = Path(f"backups/{filename}")
        if path.exists():
            with open(path, 'rb') as f:
                state = pickle.load(f)
                
            # Convert lists back to sets
            if 'visited_urls' in state:
                state['visited_urls'] = set(state['visited_urls'])
                
            if 'unique_domains' in state.get('stats', {}):
                state['stats']['unique_domains'] = set(
                    state['stats']['unique_domains']
                )
                
            # Update session state
            for key, value in state.items():
                st.session_state[key] = value
                
            return True
    except Exception as e:
        print(f"Error loading state: {e}")
        
    return False

def calculate_content_hash(content: str) -> str:
    """Calculate hash for content deduplication"""
    return hashlib.sha256(content.encode()).hexdigest()

def format_bytes(size: int) -> str:
    """Format bytes to human readable string"""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size < 1024.0:
            return f"{size:.2f} {unit}"
        size /= 1024.0
    return f"{size:.2f} PB"

def get_system_info() -> Dict[str, Any]:
    """Get system information"""
    import psutil
    import platform
    
    return {
        'platform': platform.platform(),
        'processor': platform.processor(),
        'python_version': platform.python_version(),
        'cpu_count': psutil.cpu_count(),
        'total_memory': format_bytes(psutil.virtual_memory().total),
        'available_memory': format_bytes(psutil.virtual_memory().available),
        'disk_usage': psutil.disk_usage('/').percent
    }

def validate_url(url: str) -> bool:
    """Validate URL format"""
    import re
    
    pattern = re.compile(
        r'^(https?://)?'  # http:// or https://
        r'([A-Za-z0-9.-]+)'  # domain
        r'(\.[A-Za-z]{2,})'  # .com, .org, etc
        r'(/.*)?$'  # path
    )
    
    return bool(pattern.match(url))

def generate_session_id() -> str:
    """Generate unique session ID"""
    return hashlib.sha256(
        f"{datetime.now().timestamp()}".encode()
    ).hexdigest()[:12]

def backup_database():
    """Create database backup"""
    import shutil
    from datetime import datetime
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir = Path("backups/database")
    backup_dir.mkdir(exist_ok=True)
    
    # Backup SQLite database
    if Path("data/crawler.db").exists():
        shutil.copy2("data/crawler.db", backup_dir / f"crawler_{timestamp}.db")
        
    # Backup vector database
    if Path("data/chroma_db").exists():
        shutil.copytree("data/chroma_db", backup_dir / f"chroma_{timestamp}")
        
    return timestamp

def cleanup_old_backups(days_to_keep: int = 7):
    """Clean up old backup files"""
    from datetime import datetime
    import time
    
    backup_dir = Path("backups")
    if not backup_dir.exists():
        return
        
    cutoff_time = time.time() - (days_to_keep * 24 * 60 * 60)
    
    for file_path in backup_dir.rglob("*"):
        if file_path.is_file():
            if file_path.stat().st_mtime < cutoff_time:
                try:
                    file_path.unlink()
                except:
                    pass
# core/database_manager.py
import sqlite3
from sqlite3 import Connection, Cursor
from contextlib import contextmanager
from typing import List, Dict, Any, Optional, Tuple
import json
from datetime import datetime
import pandas as pd
import numpy as np
import chromadb
from chromadb.config import Settings
import pickle
import hashlib
import zlib

class DatabaseManager:
    """Manages both SQL and vector databases"""
    
    def __init__(self, db_path: str = "data/crawler.db"):
        self.db_path = db_path
        self.init_sqlite()
        self.init_vector_db()
        
    def init_sqlite(self):
        """Initialize SQLite database with schema"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            # Create pages table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS pages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    url TEXT UNIQUE NOT NULL,
                    title TEXT,
                    description TEXT,
                    content_hash TEXT,
                    relevance_score REAL,
                    domain TEXT,
                    niche TEXT,
                    metadata JSON,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    indexed BOOLEAN DEFAULT FALSE
                )
            ''')
            
            # Create links table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS links (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    source_url TEXT,
                    target_url TEXT,
                    link_text TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (source_url) REFERENCES pages(url),
                    FOREIGN KEY (target_url) REFERENCES pages(url)
                )
            ''')
            
            # Create niches table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS niches (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT UNIQUE NOT NULL,
                    keywords JSON,
                    config JSON,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Create crawl_stats table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS crawl_stats (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT,
                    total_pages INTEGER,
                    unique_domains INTEGER,
                    avg_relevance REAL,
                    duration_seconds INTEGER,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Create indexes
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_url ON pages(url)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_domain ON pages(domain)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_niche ON pages(niche)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_relevance ON pages(relevance_score)')
            
            conn.commit()
            
    def init_vector_db(self):
        """Initialize Chroma vector database"""
        self.chroma_client = chromadb.Client(Settings(
            chroma_db_impl="duckdb+parquet",
            persist_directory="data/chroma_db"
        ))
        
        # Create or get collection
        self.vector_collection = self.chroma_client.get_or_create_collection(
            name="page_embeddings",
            metadata={"hnsw:space": "cosine"}
        )
        
    @contextmanager
    def get_connection(self):
        """Get database connection with context manager"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()
            
    def save_page(self, page_data: Dict[str, Any]) -> bool:
        """Save page data to database"""
        try:
            # Extract domain from URL
            from urllib.parse import urlparse
            domain = urlparse(page_data['url']).netloc
            
            # Create content hash for deduplication
            content_hash = hashlib.sha256(
                page_data.get('metadata', {}).get('text_content', '').encode()
            ).hexdigest()
            
            # Check for duplicates
            if self.is_duplicate(content_hash, page_data['url']):
                return False
                
            with self.get_connection() as conn:
                cursor = conn.cursor()
                
                cursor.execute('''
                    INSERT OR REPLACE INTO pages 
                    (url, title, description, content_hash, relevance_score, domain, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (
                    page_data['url'],
                    page_data.get('metadata', {}).get('title', ''),
                    page_data.get('metadata', {}).get('description', ''),
                    content_hash,
                    page_data.get('relevance', 0.0),
                    domain,
                    json.dumps(page_data.get('metadata', {}))
                ))
                
                # Save embedding to vector DB
                if 'embedding' in page_data and page_data['embedding'] is not None:
                    self.vector_collection.add(
                        embeddings=[page_data['embedding']],
                        documents=[page_data.get('metadata', {}).get('text_content', '')[:1000]],
                        metadatas=[{
                            'url': page_data['url'],
                            'domain': domain,
                            'relevance': page_data.get('relevance', 0.0),
                            'timestamp': datetime.now().isoformat()
                        }],
                        ids=[str(cursor.lastrowid)]
                    )
                    
                return True
                
        except Exception as e:
            print(f"Error saving page: {str(e)}")
            return False
            
    def is_duplicate(self, content_hash: str, url: str) -> bool:
        """Check if content is duplicate"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                'SELECT id FROM pages WHERE content_hash = ? OR url = ?',
                (content_hash, url)
            )
            return cursor.fetchone() is not None
            
    def search_content(self, query: str, filter_type: str = "All", 
                      sort_by: str = "Relevance") -> List[Dict]:
        """Search indexed content"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            # Build query based on filter type
            if filter_type == "High Relevance":
                where_clause = "WHERE relevance_score > 0.7"
            elif filter_type == "Unreviewed":
                where_clause = "WHERE indexed = FALSE"
            elif filter_type == "Flagged":
                where_clause = "WHERE relevance_score < 0.3"
            else:
                where_clause = ""
                
            # Build sort clause
            if sort_by == "Relevance":
                order_clause = "ORDER BY relevance_score DESC"
            elif sort_by == "Date":
                order_clause = "ORDER BY created_at DESC"
            elif sort_by == "Domain":
                order_clause = "ORDER BY domain"
            else:
                order_clause = ""
                
            # Execute search
            if query:
                cursor.execute(f'''
                    SELECT * FROM pages 
                    WHERE (title LIKE ? OR description LIKE ?) 
                    {where_clause}
                    {order_clause}
                    LIMIT 100
                ''', (f'%{query}%', f'%{query}%'))
            else:
                cursor.execute(f'''
                    SELECT * FROM pages 
                    {where_clause}
                    {order_clause}
                    LIMIT 100
                ''')
                
            rows = cursor.fetchall()
            return [dict(row) for row in rows]
            
    def semantic_search(self, query_embedding: List[float], limit: int = 10) -> List[Dict]:
        """Perform semantic search using vector database"""
        results = self.vector_collection.query(
            query_embeddings=[query_embedding],
            n_results=limit,
            include=['metadatas', 'documents', 'distances']
        )
        
        return results
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get database statistics"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            stats = {}
            
            # Total pages
            cursor.execute('SELECT COUNT(*) as count FROM pages')
            stats['total_pages'] = cursor.fetchone()['count']
            
            # Unique domains
            cursor.execute('SELECT COUNT(DISTINCT domain) as count FROM pages')
            stats['unique_domains'] = cursor.fetchone()['count']
            
            # Average relevance
            cursor.execute('SELECT AVG(relevance_score) as avg FROM pages')
            stats['avg_relevance'] = cursor.fetchone()['avg'] or 0.0
            
            # Recent activity
            cursor.execute('''
                SELECT COUNT(*) as count 
                FROM pages 
                WHERE created_at > datetime('now', '-1 day')
            ''')
            stats['last_24h'] = cursor.fetchone()['count']
            
            return stats
# core/ai_processor.py
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import DBSCAN, KMeans
from sklearn.decomposition import PCA
import numpy as np
from typing import List, Dict, Tuple, Any
import re
from collections import Counter
from datetime import datetime, timedelta
import networkx as nx
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk
from gensim import corpora, models
import gensim

# Download NLTK data if needed
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')

class AIProcessor:
    """Handles AI/ML processing for content analysis"""
    
    def __init__(self):
        # Load embedding model
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Initialize NLP components
        self.stop_words = set(stopwords.words('english'))
        
        # Cache for embeddings
        self.embedding_cache = {}
        
    def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for text"""
        if not text:
            return [0.0] * 384  # Default dimension
            
        # Check cache
        text_hash = hash(text)
        if text_hash in self.embedding_cache:
            return self.embedding_cache[text_hash]
            
        # Generate embedding
        embedding = self.embedding_model.encode(text).tolist()
        
        # Cache for future use
        self.embedding_cache[text_hash] = embedding
        
        return embedding
    
    def calculate_similarity(self, embedding1: List[float], 
                           embedding2: List[float]) -> float:
        """Calculate cosine similarity between embeddings"""
        if not embedding1 or not embedding2:
            return 0.0
            
        vec1 = np.array(embedding1)
        vec2 = np.array(embedding2)
        
        # Cosine similarity
        similarity = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
        
        return float(similarity)
    
    def detect_semantic_drift(self, embeddings: List[List[float]], 
                            timestamps: List[datetime]) -> Dict[str, Any]:
        """Detect semantic drift over time"""
        if len(embeddings) < 2:
            return {"drift_detected": False, "confidence": 0.0}
            
        # Calculate centroids over time windows
        embeddings_array = np.array(embeddings)
        
        # Use PCA for dimensionality reduction
        pca = PCA(n_components=2)
        reduced = pca.fit_transform(embeddings_array)
        
        # Calculate drift as distance between early and late centroids
        split_point = len(embeddings) // 2
        early_centroid = np.mean(reduced[:split_point], axis=0)
        late_centroid = np.mean(reduced[split_point:], axis=0)
        
        drift_distance = np.linalg.norm(early_centroid - late_centroid)
        
        return {
            "drift_detected": drift_distance > 0.5,
            "confidence": float(drift_distance),
            "early_centroid": early_centroid.tolist(),
            "late_centroid": late_centroid.tolist(),
            "drift_distance": float(drift_distance)
        }
    
    def cluster_content(self, embeddings: List[List[float]], 
                       n_clusters: int = 5) -> Dict[str, Any]:
        """Cluster content using KMeans"""
        if len(embeddings) < n_clusters:
            return {"clusters": [], "labels": []}
            
        embeddings_array = np.array(embeddings)
        
        # Apply KMeans clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        labels = kmeans.fit_predict(embeddings_array)
        
        # Get cluster centers
        centers = kmeans.cluster_centers_
        
        # Calculate cluster sizes
        cluster_sizes = np.bincount(labels)
        
        return {
            "clusters": centers.tolist(),
            "labels": labels.tolist(),
            "cluster_sizes": cluster_sizes.tolist(),
            "inertia": float(kmeans.inertia_)
        }
    
    def extract_keywords(self, texts: List[str], 
                        top_n: int = 10) -> List[Tuple[str, float]]:
        """Extract keywords using TF-IDF"""
        if not texts:
            return []
            
        # Create TF-IDF vectorizer
        vectorizer = TfidfVectorizer(
            max_features=100,
            stop_words='english',
            ngram_range=(1, 2)  # Include unigrams and bigrams
        )
        
        # Fit and transform
        tfidf_matrix = vectorizer.fit_transform(texts)
        
        # Get feature names
        feature_names = vectorizer.get_feature_names_out()
        
        # Sum TF-IDF scores across documents
        scores = tfidf_matrix.sum(axis=0).A1
        
        # Create list of (keyword, score) pairs
        keyword_scores = list(zip(feature_names, scores))
        
        # Sort by score
        keyword_scores.sort(key=lambda x: x[1], reverse=True)
        
        return keyword_scores[:top_n]
    
    def detect_content_gaps(self, indexed_content: List[Dict], 
                          niche_keywords: List[str]) -> List[Dict]:
        """Detect gaps in indexed content"""
        # Extract all topics from indexed content
        indexed_topics = set()
        for content in indexed_content:
            # Extract topics from content (simplified)
            text = content.get('text', '').lower()
            for keyword in niche_keywords:
                if keyword.lower() in text:
                    indexed_topics.add(keyword.lower())
                    
        # Find missing topics
        all_topics = set(kw.lower() for kw in niche_keywords)
        missing_topics = all_topics - indexed_topics
        
        return [
            {
                "topic": topic,
                "suggested_searches": self.generate_search_queries(topic),
                "estimated_volume": "medium",  # This would come from external API
                "competition": "low"
            }
            for topic in missing_topics
        ]
    
    def generate_search_queries(self, topic: str) -> List[str]:
        """Generate search queries for a topic"""
        queries = [
            f"what is {topic}",
            f"{topic} guide",
            f"best {topic} 2024",
            f"how to {topic}",
            f"{topic} tutorial",
            f"{topic} for beginners",
            f"advanced {topic}",
            f"{topic} examples",
            f"{topic} tools",
            f"{topic} resources"
        ]
        return queries
    
    def predict_trending_topics(self, historical_data: List[Dict], 
                              window_days: int = 30) -> List[Dict]:
        """Predict trending topics based on historical data"""
        # Group by date and topic
        from collections import defaultdict
        
        topic_counts = defaultdict(lambda: defaultdict(int))
        
        for data in historical_data:
            date = data['date']
            topics = data.get('topics', [])
            for topic in topics:
                topic_counts[topic][date] += 1
                
        # Calculate growth rates
        trending_topics = []
        for topic, counts in topic_counts.items():
            dates = sorted(counts.keys())
            if len(dates) >= 2:
                recent = sum(counts[d] for d in dates[-7:])  # Last 7 days
                previous = sum(counts[d] for d in dates[-14:-7])  # Previous 7 days
                
                if previous > 0:
                    growth_rate = (recent - previous) / previous
                    
                    if growth_rate > 0.5:  # 50% growth threshold
                        trending_topics.append({
                            "topic": topic,
                            "growth_rate": growth_rate,
                            "recent_mentions": recent,
                            "previous_mentions": previous
                        })
                        
        # Sort by growth rate
        trending_topics.sort(key=lambda x: x['growth_rate'], reverse=True)
        
        return trending_topics[:10]
# core/crawler_manager.py
import asyncio
import aiohttp
from bs4 import BeautifulSoup
from urllib.parse import urlparse, urljoin
import time
from typing import List, Dict, Set, Optional
import re
from datetime import datetime
import logging
from concurrent.futures import ThreadPoolExecutor
import backoff
from tenacity import retry, stop_after_attempt, wait_exponential
import redis
from celery import Celery

class AsyncCrawler:
    """Asynchronous web crawler with rate limiting and politeness"""
    
    def __init__(self, max_workers: int = 10, rate_limit: float = 1.0):
        self.max_workers = max_workers
        self.rate_limit = rate_limit
        self.session = None
        self.visited_urls = set()
        self.domain_timers = {}
        self.user_agent = "Mozilla/5.0 (compatible; IntelligentCrawler/1.0; +https://example.com/bot)"
        self.headers = {
            'User-Agent': self.user_agent,
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'DNT': '1',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1'
        }
        
    async def __aenter__(self):
        self.session = aiohttp.ClientSession(headers=self.headers)
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
            
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def fetch_url(self, url: str) -> Optional[str]:
        """Fetch URL with retry logic and rate limiting"""
        domain = urlparse(url).netloc
        
        # Rate limiting per domain
        if domain in self.domain_timers:
            elapsed = time.time() - self.domain_timers[domain]
            if elapsed < self.rate_limit:
                await asyncio.sleep(self.rate_limit - elapsed)
                
        try:
            async with self.session.get(url, timeout=10, ssl=False) as response:
                self.domain_timers[domain] = time.time()
                
                if response.status == 200:
                    return await response.text()
                elif response.status == 429:  # Too Many Requests
                    await asyncio.sleep(60)  # Wait 1 minute
                    raise Exception("Rate limited")
                else:
                    logging.warning(f"Failed to fetch {url}: Status {response.status}")
                    return None
        except Exception as e:
            logging.error(f"Error fetching {url}: {str(e)}")
            raise
            
    def extract_links(self, html: str, base_url: str) -> List[str]:
        """Extract all links from HTML content"""
        soup = BeautifulSoup(html, 'lxml')
        links = []
        
        for link in soup.find_all('a', href=True):
            href = link['href']
            # Convert relative URLs to absolute
            absolute_url = urljoin(base_url, href)
            # Filter out non-HTTP links and common non-content links
            if absolute_url.startswith(('http://', 'https://')):
                # Skip common non-content URLs
                if not any(skip in absolute_url.lower() for skip in [
                    '.pdf', '.jpg', '.png', '.gif', '.zip', '.exe', 
                    'mailto:', 'tel:', 'javascript:'
                ]):
                    links.append(absolute_url)
                    
        return links
    
    def filter_by_niche(self, text: str, niche_keywords: List[str]) -> float:
        """Calculate niche relevance score for text"""
        if not text:
            return 0.0
            
        text_lower = text.lower()
        matches = 0
        total_keywords = len(niche_keywords)
        
        for keyword in niche_keywords:
            if keyword.lower() in text_lower:
                matches += 1
                
        return matches / total_keywords if total_keywords > 0 else 0.0

class CrawlerManager:
    """Manages crawling operations and worker coordination"""
    
    def __init__(self):
        self.crawler = None
        self.is_crawling = False
        self.is_paused = False
        self.workers = []
        self.task_queue = asyncio.Queue()
        self.results_queue = asyncio.Queue()
        self.redis_client = redis.Redis(host='localhost', port=6379, db=0)
        self.celery_app = Celery('crawler', broker='redis://localhost:6379/0')
        
    def start_crawling(self, niche_profile: Dict, workers: int = 5):
        """Start the crawling process"""
        self.is_crawling = True
        self.is_paused = False
        
        # Initialize crawler
        self.crawler = AsyncCrawler(max_workers=workers)
        
        # Start workers
        for i in range(workers):
            worker = asyncio.create_task(self.worker_task(i, niche_profile))
            self.workers.append(worker)
            
        # Start result processor
        asyncio.create_task(self.process_results())
        
    async def worker_task(self, worker_id: int, niche_profile: Dict):
        """Worker task for crawling URLs"""
        async with self.crawler as crawler:
            while self.is_crawling and not self.is_paused:
                try:
                    # Get URL from queue with timeout
                    url = await asyncio.wait_for(self.task_queue.get(), timeout=1.0)
                    
                    # Fetch URL
                    html = await crawler.fetch_url(url)
                    
                    if html:
                        # Calculate relevance
                        relevance = crawler.filter_by_niche(html, niche_profile['keywords'])
                        
                        # Extract links
                        if relevance > niche_profile.get('threshold', 0.3):
                            links = crawler.extract_links(html, url)
                            for link in links:
                                await self.task_queue.put(link)
                                
                        # Send to results queue
                        await self.results_queue.put({
                            'url': url,
                            'relevance': relevance,
                            'html': html[:10000],  # Store first 10k chars
                            'timestamp': datetime.now().isoformat(),
                            'worker_id': worker_id
                        })
                        
                    self.task_queue.task_done()
                    
                except asyncio.TimeoutError:
                    continue
                except Exception as e:
                    logging.error(f"Worker {worker_id} error: {str(e)}")
                    continue
                    
    async def process_results(self):
        """Process crawled results"""
        from core.database_manager import DatabaseManager
        from core.ai_processor import AIProcessor
        
        db_manager = DatabaseManager()
        ai_processor = AIProcessor()
        
        while self.is_crawling:
            try:
                result = await asyncio.wait_for(self.results_queue.get(), timeout=1.0)
                
                # Extract metadata
                soup = BeautifulSoup(result['html'], 'lxml')
                metadata = {
                    'title': soup.title.string if soup.title else '',
                    'description': soup.find('meta', attrs={'name': 'description'})['content'] 
                                 if soup.find('meta', attrs={'name': 'description'}) else '',
                    'keywords': soup.find('meta', attrs={'name': 'keywords'})['content']
                               if soup.find('meta', attrs={'name': 'keywords'}) else '',
                    'text_content': soup.get_text()[:5000]  # First 5000 chars
                }
                
                # Generate embedding
                embedding = ai_processor.generate_embedding(metadata['text_content'])
                
                # Store in database
                db_manager.save_page({
                    'url': result['url'],
                    'relevance': result['relevance'],
                    'metadata': metadata,
                    'embedding': embedding,
                    'timestamp': result['timestamp']
                })
                
                self.results_queue.task_done()
                
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logging.error(f"Error processing result: {str(e)}")
                
    def pause_crawling(self):
        """Pause all crawling activities"""
        self.is_paused = True
        
    def resume_crawling(self):
        """Resume crawling"""
        self.is_paused = False
        
    def stop_crawling(self):
        """Stop all crawling activities"""
        self.is_crawling = False
        self.is_paused = False
        
        # Cancel all worker tasks
        for worker in self.workers:
            worker.cancel()
        self.workers.clear()
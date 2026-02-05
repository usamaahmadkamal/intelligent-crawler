# setup.py
import sys
import subprocess
import platform
from pathlib import Path

def check_and_install_packages():
    """Check and install required packages"""
    required_packages = [
        'streamlit>=1.28.0',
        'beautifulsoup4>=4.12.0',
        'requests>=2.31.0',
        'aiohttp>=3.9.0',
        'pandas>=2.0.0',
        'numpy>=1.24.0',
        'scikit-learn>=1.3.0',
        'sentence-transformers>=2.2.0',
        'chromadb>=0.4.0',
        'sqlalchemy>=2.0.0',
        'lxml>=4.9.0',
        'plotly>=5.17.0',
        'python-dotenv>=1.0.0',
        'redis>=5.0.0',
        'celery>=5.3.0',
        'psutil>=5.9.0',
        'nltk>=3.8.0',
        'gensim>=4.3.0',
        'networkx>=3.0'
    ]
    
    print("Checking and installing required packages...")
    
    for package in required_packages:
        try:
            __import__(package.split('>=')[0].split('==')[0])
            print(f"✓ {package} already installed")
        except ImportError:
            print(f"Installing {package}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            
def setup_directories():
    """Create necessary directories"""
    directories = ['data', 'logs', 'exports', 'models', 'configs', 'backups', 'chroma_db']
    
    for dir_name in directories:
        Path(dir_name).mkdir(exist_ok=True)
        print(f"Created directory: {dir_name}")
        
def setup_database():
    """Initialize database"""
    from core.database_manager import DatabaseManager
    
    try:
        db_manager = DatabaseManager()
        print("✓ Database initialized successfully")
        return True
    except Exception as e:
        print(f"✗ Database initialization failed: {e}")
        return False
        
def download_nltk_data():
    """Download required NLTK data"""
    import nltk
    
    required_data = ['punkt', 'stopwords', 'averaged_perceptron_tagger']
    
    for data in required_data:
        try:
            nltk.data.find(f'tokenizers/{data}')
            print(f"✓ NLTK {data} already downloaded")
        except LookupError:
            print(f"Downloading NLTK {data}...")
            nltk.download(data)
            
def check_redis():
    """Check if Redis is running"""
    import redis
    
    try:
        r = redis.Redis(host='localhost', port=6379, db=0)
        r.ping()
        print("✓ Redis is running")
        return True
    except:
        print("⚠ Redis is not running. Some features may be limited.")
        print("  To install Redis:")
        print("  - macOS: brew install redis")
        print("  - Ubuntu: sudo apt install redis-server")
        print("  - Windows: Download from https://github.com/microsoftarchive/redis/releases")
        return False
        
def create_env_file():
    """Create .env file if it doesn't exist"""
    env_path = Path(".env")
    
    if not env_path.exists():
        env_template = """# Database settings
DATABASE_URL=sqlite:///data/crawler.db
REDIS_URL=redis://localhost:6379/0

# Crawling settings
DEFAULT_WORKERS=5
RATE_LIMIT=1.0
MAX_DEPTH=3
TIMEOUT=10

# AI/ML settings
EMBEDDING_MODEL=all-MiniLM-L6-v2
SIMILARITY_THRESHOLD=0.7

# API Keys (optional)
GOOGLE_API_KEY=your_google_api_key_here
GOOGLE_CSE_ID=your_cse_id_here

# Notification settings
ENABLE_NOTIFICATIONS=False
EMAIL_NOTIFICATIONS=False
SLACK_WEBHOOK=your_slack_webhook_here

# Security settings
USER_AGENT=IntelligentCrawler/1.0 (+https://example.com/bot)
RESPECT_ROBOTS=True

# Performance settings
MAX_QUEUE_SIZE=10000
CACHE_SIZE=1000
BATCH_SIZE=100

# UI settings
THEME=auto
REFRESH_INTERVAL=5
"""
        
        with open(env_path, 'w') as f:
            f.write(env_template)
            
        print("Created .env file with default settings")
        
def main():
    """Main setup function"""
    print("=" * 50)
    print("Intelligent Web Crawling Platform - Setup")
    print("=" * 50)
    
    # Step 1: Check Python version
    if sys.version_info < (3, 8):
        print("✗ Python 3.8 or higher is required")
        sys.exit(1)
        
    print(f"✓ Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")
    
    # Step 2: Install packages
    check_and_install_packages()
    
    # Step 3: Create directories
    setup_directories()
    
    # Step 4: Setup database
    setup_database()
    
    # Step 5: Download NLTK data
    download_nltk_data()
    
    # Step 6: Check Redis
    check_redis()
    
    # Step 7: Create .env file
    create_env_file()
    
    print("\n" + "=" * 50)
    print("Setup completed successfully!")
    print("\nTo start the application:")
    print("1. streamlit run main.py")
    print("\nTo run with custom settings:")
    print("2. streamlit run main.py --server.port 8501 --server.headless true")
    print("=" * 50)
    
if __name__ == "__main__":
    main()
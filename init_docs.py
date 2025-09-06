"""Initialize document sources for EchoCheck."""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from data.loader import DocumentLoader
from utils.config import config
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def setup_document_sources():
    """Setup and initialize document sources."""
    print("🔄 Initializing EchoCheck document sources...")
    
    # Create loader
    loader = DocumentLoader()
    
    # Create subdirectories for different types
    (loader.sources_dir / "react").mkdir(exist_ok=True)
    (loader.sources_dir / "nextjs").mkdir(exist_ok=True)
    (loader.sources_dir / "custom").mkdir(exist_ok=True)
    
    print("📁 Created source directories")
    print(f"  - React docs: {loader.sources_dir / 'react'}")
    print(f"  - Next.js docs: {loader.sources_dir / 'nextjs'}")
    print(f"  - Custom docs: {loader.sources_dir / 'custom'}")
    
    # Load all documents
    print("\n📚 Loading documents...")
    docs = loader.load_all_documents(mode="live_only")
    
    print(f"\n✅ Loaded {len(docs)} documents from various sources")
    
    # Get cache info
    cache_info = loader.get_cache_info()
    if cache_info.get("exists"):
        print("\n📊 Document sources:")
        
        # Count by source
        source_counts = {}
        for doc in docs:
            source = doc.metadata.get("source", "unknown")
            source_counts[source] = source_counts.get(source, 0) + 1
        
        for source, count in source_counts.items():
            print(f"  - {source}: {count} documents")
        
        print(f"\n💾 Documents cached to: {loader.sources_dir / 'cached_docs.json'}")
        print(f"🕒 Cache expires in: {config.CACHE_EXPIRY_HOURS} hours")
    
    print("\n🎉 Document initialization complete!")
    return docs

def show_config():
    """Show current configuration."""
    print("⚙️ Current Configuration:")
    print(f"  - Document Source Mode: {config.DOCUMENT_SOURCE_MODE}")
    print(f"  - Use Cache: {config.USE_DOCUMENT_CACHE}")
    print(f"  - Cache Expiry: {config.CACHE_EXPIRY_HOURS} hours")
    print(f"  - React Docs: {'✅' if config.ENABLE_REACT_DOCS else '❌'}")
    print(f"  - Next.js Docs: {'✅' if config.ENABLE_NEXTJS_DOCS else '❌'}")
    print(f"  - Local Docs: {'✅' if config.ENABLE_LOCAL_DOCS else '❌'}")
    print(f"  - Sample Docs: {'✅' if config.ENABLE_SAMPLE_DOCS else '❌'}")

def main():
    """Main initialization function."""
    print("🛡️ EchoCheck Document Initialization")
    print("=" * 50)
    
    # Show configuration
    show_config()
    print()
    
    # Setup documents
    docs = setup_document_sources()
    
    print("\n" + "=" * 50)
    print("🚀 Ready to run EchoCheck!")
    print("Run: streamlit run app.py")

if __name__ == "__main__":
    main()

"""
Script to check if metrics are being saved to the database.
Run: python scripts/check_metrics.py
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from monitoring.metrics_store import get_metrics_store

def main():
    store = get_metrics_store()
    
    print("Checking metrics database...")
    print(f"Database path: {store.db_path}")
    
    # Get all metrics without filter
    all_metrics = store.get_query_metrics(limit=10)
    
    print(f"\nTotal metrics found: {len(all_metrics)}")
    
    if all_metrics:
        print("\nRecent metrics:")
        for i, metric in enumerate(all_metrics[:5], 1):
            print(f"\n{i}. Query ID: {metric.get('id', 'N/A')}")
            print(f"   Query: {metric.get('query', 'N/A')[:50]}...")
            print(f"   Total Latency: {metric.get('total_latency_ms', 'N/A')} ms")
            print(f"   Created At: {metric.get('created_at', 'N/A')}")
    else:
        print("\n⚠️ No metrics found in database!")
        print("This could mean:")
        print("1. No queries have been executed yet")
        print("2. Metrics collection is not working")
        print("3. Database path is incorrect")
    
    # Check latency stats
    latency_stats = store.get_latency_stats()
    print(f"\nLatency Stats: {latency_stats}")
    
    # Check retrieval stats
    retrieval_stats = store.get_retrieval_stats()
    print(f"\nRetrieval Stats: {retrieval_stats}")

if __name__ == "__main__":
    main()


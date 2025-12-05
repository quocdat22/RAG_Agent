"""
Metrics storage using SQLite database.
Stores query metrics, retrieval metrics, and chunk quality data.
"""

import json
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional

from config.settings import get_settings


class MetricsStore:
    """Manages metrics storage in SQLite database."""

    def __init__(self, db_path: Optional[str] = None):
        settings = get_settings()
        self.db_path = db_path or getattr(
            settings, "metrics_db_path", "./data/metrics.db"
        )
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        """Get a database connection."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self):
        """Initialize database tables if they don't exist."""
        conn = self._get_connection()
        try:
            # Query metrics table - tracks overall query performance
            conn.execute("""
                CREATE TABLE IF NOT EXISTS query_metrics (
                    id TEXT PRIMARY KEY,
                    query TEXT NOT NULL,
                    conversation_id TEXT,
                    total_latency_ms REAL NOT NULL,
                    retrieval_latency_ms REAL,
                    reranking_latency_ms REAL,
                    llm_latency_ms REAL,
                    num_candidates INTEGER,
                    num_final_chunks INTEGER,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Retrieval metrics table - tracks retrieval quality
            conn.execute("""
                CREATE TABLE IF NOT EXISTS retrieval_metrics (
                    id TEXT PRIMARY KEY,
                    query_metric_id TEXT NOT NULL,
                    retrieval_method TEXT NOT NULL,
                    num_results INTEGER NOT NULL,
                    avg_score REAL,
                    min_score REAL,
                    max_score REAL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (query_metric_id) REFERENCES query_metrics(id) ON DELETE CASCADE
                )
            """)

            # Chunk quality table - tracks individual chunk quality
            conn.execute("""
                CREATE TABLE IF NOT EXISTS chunk_quality (
                    id TEXT PRIMARY KEY,
                    query_metric_id TEXT NOT NULL,
                    chunk_index INTEGER NOT NULL,
                    chunk_text_preview TEXT,
                    score REAL NOT NULL,
                    file_path TEXT,
                    chunk_index_in_doc INTEGER,
                    retrieval_method TEXT,
                    is_final_chunk INTEGER DEFAULT 0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (query_metric_id) REFERENCES query_metrics(id) ON DELETE CASCADE
                )
            """)

            # Precision/Recall metrics (requires ground truth - optional)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS precision_recall_metrics (
                    id TEXT PRIMARY KEY,
                    query_metric_id TEXT NOT NULL,
                    relevant_chunks TEXT,
                    retrieved_chunks TEXT,
                    precision REAL,
                    recall REAL,
                    f1_score REAL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (query_metric_id) REFERENCES query_metrics(id) ON DELETE CASCADE
                )
            """)

            # Create indexes for faster queries
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_query_metrics_created 
                ON query_metrics(created_at)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_query_metrics_conversation 
                ON query_metrics(conversation_id)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_retrieval_metrics_query 
                ON retrieval_metrics(query_metric_id)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_chunk_quality_query 
                ON chunk_quality(query_metric_id)
            """)

            conn.commit()
        finally:
            conn.close()

    def save_query_metric(
        self,
        query_id: str,
        query: str,
        total_latency_ms: float,
        conversation_id: Optional[str] = None,
        retrieval_latency_ms: Optional[float] = None,
        reranking_latency_ms: Optional[float] = None,
        llm_latency_ms: Optional[float] = None,
        num_candidates: Optional[int] = None,
        num_final_chunks: Optional[int] = None,
    ):
        """Save query-level metrics."""
        conn = self._get_connection()
        try:
            conn.execute("""
                INSERT INTO query_metrics (
                    id, query, conversation_id, total_latency_ms,
                    retrieval_latency_ms, reranking_latency_ms, llm_latency_ms,
                    num_candidates, num_final_chunks
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                query_id,
                query,
                conversation_id,
                total_latency_ms,
                retrieval_latency_ms,
                reranking_latency_ms,
                llm_latency_ms,
                num_candidates,
                num_final_chunks,
            ))
            conn.commit()
        finally:
            conn.close()

    def save_retrieval_metric(
        self,
        retrieval_id: str,
        query_metric_id: str,
        retrieval_method: str,
        num_results: int,
        avg_score: Optional[float] = None,
        min_score: Optional[float] = None,
        max_score: Optional[float] = None,
    ):
        """Save retrieval-level metrics."""
        conn = self._get_connection()
        try:
            conn.execute("""
                INSERT INTO retrieval_metrics (
                    id, query_metric_id, retrieval_method,
                    num_results, avg_score, min_score, max_score
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                retrieval_id,
                query_metric_id,
                retrieval_method,
                num_results,
                avg_score,
                min_score,
                max_score,
            ))
            conn.commit()
        finally:
            conn.close()

    def save_chunk_quality(
        self,
        chunk_id: str,
        query_metric_id: str,
        chunk_index: int,
        score: float,
        chunk_text_preview: Optional[str] = None,
        file_path: Optional[str] = None,
        chunk_index_in_doc: Optional[int] = None,
        retrieval_method: Optional[str] = None,
        is_final_chunk: bool = False,
    ):
        """Save chunk-level quality metrics."""
        conn = self._get_connection()
        try:
            # Truncate preview if too long
            if chunk_text_preview and len(chunk_text_preview) > 500:
                chunk_text_preview = chunk_text_preview[:500] + "..."
            
            conn.execute("""
                INSERT INTO chunk_quality (
                    id, query_metric_id, chunk_index, chunk_text_preview,
                    score, file_path, chunk_index_in_doc, retrieval_method, is_final_chunk
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                chunk_id,
                query_metric_id,
                chunk_index,
                chunk_text_preview,
                score,
                file_path,
                chunk_index_in_doc,
                retrieval_method,
                1 if is_final_chunk else 0,
            ))
            conn.commit()
        finally:
            conn.close()

    def save_precision_recall(
        self,
        pr_id: str,
        query_metric_id: str,
        relevant_chunks: List[str],
        retrieved_chunks: List[str],
        precision: float,
        recall: float,
        f1_score: float,
    ):
        """Save precision/recall metrics (requires ground truth)."""
        conn = self._get_connection()
        try:
            conn.execute("""
                INSERT INTO precision_recall_metrics (
                    id, query_metric_id, relevant_chunks, retrieved_chunks,
                    precision, recall, f1_score
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                pr_id,
                query_metric_id,
                json.dumps(relevant_chunks),
                json.dumps(retrieved_chunks),
                precision,
                recall,
                f1_score,
            ))
            conn.commit()
        finally:
            conn.close()

    def get_query_metrics(
        self,
        limit: int = 100,
        offset: int = 0,
        conversation_id: Optional[str] = None,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """Get query metrics with optional filters."""
        conn = self._get_connection()
        try:
            query = "SELECT * FROM query_metrics WHERE 1=1"
            params = []

            if conversation_id:
                query += " AND conversation_id = ?"
                params.append(conversation_id)

            if start_time:
                # Convert timestamp to datetime string for SQLite comparison
                from datetime import datetime
                start_dt = datetime.fromtimestamp(start_time)
                query += " AND created_at >= ?"
                params.append(start_dt.strftime("%Y-%m-%d %H:%M:%S"))

            if end_time:
                # Convert timestamp to datetime string for SQLite comparison
                from datetime import datetime
                end_dt = datetime.fromtimestamp(end_time)
                query += " AND created_at <= ?"
                params.append(end_dt.strftime("%Y-%m-%d %H:%M:%S"))

            query += " ORDER BY created_at DESC LIMIT ? OFFSET ?"
            params.extend([limit, offset])

            cursor = conn.execute(query, params)
            rows = cursor.fetchall()
            return [dict(row) for row in rows]
        finally:
            conn.close()

    def get_retrieval_stats(
        self,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Get aggregated retrieval statistics."""
        conn = self._get_connection()
        try:
            query = """
                SELECT 
                    retrieval_method,
                    COUNT(*) as count,
                    AVG(num_results) as avg_num_results,
                    AVG(avg_score) as avg_score,
                    AVG(min_score) as avg_min_score,
                    AVG(max_score) as avg_max_score
                FROM retrieval_metrics rm
                JOIN query_metrics qm ON rm.query_metric_id = qm.id
                WHERE 1=1
            """
            params = []

            if start_time:
                from datetime import datetime
                start_dt = datetime.fromtimestamp(start_time)
                query += " AND qm.created_at >= ?"
                params.append(start_dt.strftime("%Y-%m-%d %H:%M:%S"))

            if end_time:
                from datetime import datetime
                end_dt = datetime.fromtimestamp(end_time)
                query += " AND qm.created_at <= ?"
                params.append(end_dt.strftime("%Y-%m-%d %H:%M:%S"))

            query += " GROUP BY retrieval_method"

            cursor = conn.execute(query, params)
            rows = cursor.fetchall()
            return [dict(row) for row in rows]
        finally:
            conn.close()

    def get_chunk_quality_stats(
        self,
        query_metric_id: Optional[str] = None,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Get aggregated chunk quality statistics."""
        conn = self._get_connection()
        try:
            query = """
                SELECT 
                    AVG(score) as avg_score,
                    MIN(score) as min_score,
                    MAX(score) as max_score,
                    COUNT(*) as total_chunks,
                    SUM(is_final_chunk) as final_chunks
                FROM chunk_quality cq
                JOIN query_metrics qm ON cq.query_metric_id = qm.id
                WHERE 1=1
            """
            params = []

            if query_metric_id:
                query += " AND cq.query_metric_id = ?"
                params.append(query_metric_id)

            if start_time:
                from datetime import datetime
                start_dt = datetime.fromtimestamp(start_time)
                query += " AND qm.created_at >= ?"
                params.append(start_dt.strftime("%Y-%m-%d %H:%M:%S"))

            if end_time:
                from datetime import datetime
                end_dt = datetime.fromtimestamp(end_time)
                query += " AND qm.created_at <= ?"
                params.append(end_dt.strftime("%Y-%m-%d %H:%M:%S"))

            cursor = conn.execute(query, params)
            row = cursor.fetchone()
            return dict(row) if row else {}
        finally:
            conn.close()

    def get_latency_stats(
        self,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Get aggregated latency statistics."""
        conn = self._get_connection()
        try:
            query = """
                SELECT 
                    AVG(total_latency_ms) as avg_total_latency,
                    MIN(total_latency_ms) as min_total_latency,
                    MAX(total_latency_ms) as max_total_latency,
                    AVG(retrieval_latency_ms) as avg_retrieval_latency,
                    AVG(reranking_latency_ms) as avg_reranking_latency,
                    AVG(llm_latency_ms) as avg_llm_latency,
                    COUNT(*) as total_queries
                FROM query_metrics
                WHERE 1=1
            """
            params = []

            if start_time:
                from datetime import datetime
                start_dt = datetime.fromtimestamp(start_time)
                query += " AND created_at >= ?"
                params.append(start_dt.strftime("%Y-%m-%d %H:%M:%S"))

            if end_time:
                from datetime import datetime
                end_dt = datetime.fromtimestamp(end_time)
                query += " AND created_at <= ?"
                params.append(end_dt.strftime("%Y-%m-%d %H:%M:%S"))

            cursor = conn.execute(query, params)
            row = cursor.fetchone()
            return dict(row) if row else {}
        finally:
            conn.close()


# Singleton instance
_metrics_store_instance: Optional[MetricsStore] = None


def get_metrics_store(reload: bool = False) -> MetricsStore:
    """Get metrics store instance."""
    global _metrics_store_instance
    if reload or _metrics_store_instance is None:
        _metrics_store_instance = MetricsStore()
    return _metrics_store_instance


# Monitoring & Observability Module

Module này cung cấp hệ thống monitoring và observability cho RAG Agent, giải quyết các vấn đề:

- ✅ **Query Latency Tracking**: Track latency cho toàn bộ query và từng phase (retrieval, reranking, LLM)
- ✅ **Retrieved Chunks Quality Logging**: Log quality metrics cho từng chunk được retrieve (scores, metadata)
- ✅ **Retrieval Precision/Recall Measurement**: Tính toán precision, recall, và F1 score (cần ground truth)
- ✅ **Monitoring Dashboard**: Dashboard Streamlit để visualize metrics

## Cấu trúc

- `metrics_store.py`: SQLite database để lưu trữ metrics
- `metrics_collector.py`: Collector để track và lưu metrics
- `dashboard.py`: Streamlit dashboard để visualize metrics

## Sử dụng

### 1. Metrics tự động được collect

Khi bạn gọi `answer_query()` trong `rag.pipeline`, metrics sẽ tự động được collect và lưu vào database.

### 2. Xem metrics qua API

```bash
# Get query metrics
curl -H "X-API-Key: your-key" http://localhost:8000/metrics/queries

# Get latency stats
curl -H "X-API-Key: your-key" http://localhost:8000/metrics/latency

# Get retrieval stats
curl -H "X-API-Key: your-key" http://localhost:8000/metrics/retrieval

# Get chunk quality stats
curl -H "X-API-Key: your-key" http://localhost:8000/metrics/chunks

# Get comprehensive summary
curl -H "X-API-Key: your-key" http://localhost:8000/metrics/summary
```

### 3. Chạy Dashboard

```bash
streamlit run monitoring/dashboard.py
```

Dashboard sẽ hiển thị:
- Overview metrics (total queries, avg latency, etc.)
- Query latency analysis với charts
- Retrieval quality metrics
- Chunk quality metrics
- Recent queries table

## Database Schema

### query_metrics
- `id`: Query ID
- `query`: Query text
- `conversation_id`: Conversation ID (optional)
- `total_latency_ms`: Total query latency
- `retrieval_latency_ms`: Retrieval phase latency
- `reranking_latency_ms`: Reranking phase latency
- `llm_latency_ms`: LLM generation latency
- `num_candidates`: Number of candidates retrieved
- `num_final_chunks`: Number of final chunks after reranking
- `created_at`: Timestamp

### retrieval_metrics
- `id`: Retrieval metric ID
- `query_metric_id`: Foreign key to query_metrics
- `retrieval_method`: Method used (e.g., "hybrid", "dense", "sparse")
- `num_results`: Number of results
- `avg_score`, `min_score`, `max_score`: Score statistics

### chunk_quality
- `id`: Chunk quality ID
- `query_metric_id`: Foreign key to query_metrics
- `chunk_index`: Index of chunk in results
- `chunk_text_preview`: Preview of chunk text
- `score`: Chunk score
- `file_path`: Source file path
- `chunk_index_in_doc`: Index in original document
- `retrieval_method`: Method that retrieved this chunk
- `is_final_chunk`: Whether this chunk was used in final answer

### precision_recall_metrics
- `id`: PR metric ID
- `query_metric_id`: Foreign key to query_metrics
- `relevant_chunks`: JSON array of relevant chunk IDs (ground truth)
- `retrieved_chunks`: JSON array of retrieved chunk IDs
- `precision`, `recall`, `f1_score`: Calculated metrics

## Tính toán Precision/Recall

Để tính precision/recall, bạn cần cung cấp ground truth (relevant chunks). Ví dụ:

```python
from monitoring.metrics_collector import MetricsCollector

metrics = MetricsCollector()
metrics.start_query(query, conversation_id)

# ... perform retrieval ...

# After retrieval, calculate precision/recall
relevant_chunks = ["chunk_id_1", "chunk_id_2"]  # Ground truth
retrieved_chunks = [c.id for c in candidates]  # Retrieved chunks

pr_metrics = metrics.calculate_precision_recall(
    relevant_chunks=relevant_chunks,
    retrieved_chunks=retrieved_chunks,
)
```

## Configuration

Metrics database path có thể được cấu hình trong `.env`:

```
METRICS_DB_PATH=./data/metrics.db
```

Mặc định là `./data/metrics.db`.


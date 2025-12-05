# Monitoring & Observability Implementation Summary

## Vấn đề đã được giải quyết

### ✅ 1. Query Latency Tracking
**Trước đây**: Không track query latency  
**Bây giờ**: 
- Track total latency cho mỗi query
- Track latency cho từng phase:
  - Retrieval latency
  - Reranking latency  
  - LLM generation latency
- Metrics được lưu vào database và có thể query qua API

**Implementation**:
- `MetricsCollector` class trong `monitoring/metrics_collector.py`
- Tích hợp vào `rag/pipeline.py` để tự động track mỗi query
- Lưu vào `query_metrics` table trong SQLite database

### ✅ 2. Retrieved Chunks Quality Logging
**Trước đây**: Không log quality của retrieved chunks  
**Bây giờ**:
- Log score cho mỗi chunk được retrieve
- Log metadata (file_path, chunk_index, etc.)
- Track min/max/avg scores
- Phân biệt chunks từ retrieval vs final chunks sau reranking

**Implementation**:
- `chunk_quality` table lưu thông tin chi tiết từng chunk
- `retrieval_metrics` table lưu aggregated stats
- Tự động log khi chunks được retrieve và rerank

### ✅ 3. Retrieval Precision/Recall Measurement
**Trước đây**: Không measure precision/recall  
**Bây giờ**:
- Có function `calculate_precision_recall()` để tính toán
- Lưu precision, recall, và F1 score vào database
- **Lưu ý**: Cần ground truth (relevant chunks) để tính toán chính xác

**Implementation**:
- `precision_recall_metrics` table trong database
- Method `calculate_precision_recall()` trong `MetricsCollector`
- Có thể gọi sau khi retrieval để tính toán (nếu có ground truth)

### ✅ 4. Monitoring Dashboard
**Trước đây**: Không có dashboard  
**Bây giờ**:
- Streamlit dashboard với các visualizations:
  - Overview metrics (total queries, avg latency, etc.)
  - Query latency analysis với line charts và histograms
  - Retrieval quality metrics
  - Chunk quality statistics
  - Recent queries table
- Filters theo time range và conversation ID

**Implementation**:
- `monitoring/dashboard.py` - Streamlit app
- Chạy với: `streamlit run monitoring/dashboard.py`

## Cấu trúc Files

```
monitoring/
├── __init__.py              # Module exports
├── metrics_store.py          # SQLite database operations
├── metrics_collector.py      # Metrics collection logic
├── dashboard.py              # Streamlit dashboard
└── README.md                 # Documentation

api/main.py                   # Added metrics API endpoints
rag/pipeline.py               # Integrated metrics collection
config/settings.py            # Added metrics_db_path setting
```

## API Endpoints mới

1. `GET /metrics/queries` - Get query metrics
2. `GET /metrics/latency` - Get latency statistics
3. `GET /metrics/retrieval` - Get retrieval statistics
4. `GET /metrics/chunks` - Get chunk quality statistics
5. `GET /metrics/summary` - Get comprehensive summary

## Database Schema

### query_metrics
Lưu metrics cho mỗi query:
- Total latency và latency từng phase
- Số lượng candidates và final chunks
- Query text và conversation ID

### retrieval_metrics
Lưu aggregated metrics cho retrieval:
- Retrieval method (hybrid, dense, sparse)
- Score statistics (avg, min, max)
- Number of results

### chunk_quality
Lưu quality metrics cho từng chunk:
- Chunk score
- File path và metadata
- Retrieval method
- Flag cho final chunks

### precision_recall_metrics
Lưu precision/recall metrics:
- Relevant chunks (ground truth)
- Retrieved chunks
- Calculated precision, recall, F1

## Cách sử dụng

### 1. Metrics tự động được collect
Khi gọi `answer_query()`, metrics tự động được track và lưu.

### 2. Xem qua API
```bash
curl -H "X-API-Key: your-key" http://localhost:8000/metrics/summary
```

### 3. Xem qua Dashboard
```bash
streamlit run monitoring/dashboard.py
```

## Configuration

Thêm vào `.env` (optional):
```
METRICS_DB_PATH=./data/metrics.db
```

Mặc định: `./data/metrics.db`

## Dependencies mới

- `plotly` - For charts in dashboard
- `pandas` - For data manipulation in dashboard

Đã được thêm vào `requirements.txt`.

## Testing

Để test:
1. Chạy một số queries qua API hoặc UI
2. Xem metrics qua API: `GET /metrics/summary`
3. Hoặc mở dashboard: `streamlit run monitoring/dashboard.py`

## Notes

- Precision/Recall cần ground truth để tính toán chính xác
- Metrics database sẽ tự động được tạo khi chạy lần đầu
- Dashboard cần có dữ liệu queries trước khi hiển thị metrics

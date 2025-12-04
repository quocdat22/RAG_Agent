<!-- 2da2e804-2ef4-4019-9785-79da102dd10a 420cf3ee-8471-461f-9306-affc51f54af9 -->
# Fix Vector Store Version Control Issues

## Vấn đề hiện tại

1. **Duplicate chunks khi re-ingest**: Trong [`rag/vector_store.py`](rag/vector_store.py) line 60, chunk IDs được tạo với UUID ngẫu nhiên (`f"{file_path}_{chunk_index}_{uuid.uuid4().hex[:8]}"`), khiến mỗi lần re-ingest tạo ra chunks mới thay vì update.

2. **Không track document version**: Không có metadata fields để track file hash, modification time, hoặc version number.

3. **Không có update mechanism**: Chỉ sử dụng `add()` method, không check hoặc update existing chunks.

## Giải pháp

### 1. Thêm Version Tracking Metadata

**File: [`ingestion/metadata_schema.py`](ingestion/metadata_schema.py)**

- Thêm các fields mới vào `MetadataFields`:
  - `FILE_HASH`: Hash của file content (SHA256)
  - `FILE_MTIME`: Modification time của file
  - `DOCUMENT_VERSION`: Version number (incremental)
- Cập nhật `DocumentMetadata` Pydantic model để include các fields này

**File: [`ingestion/pipeline.py`](ingestion/pipeline.py)**

- Tạo helper function `calculate_file_hash()` để tính SHA256 hash của file
- Tạo helper function `get_file_metadata()` để lấy file modification time
- Trong `run_ingestion()`, tính hash và mtime cho mỗi document và thêm vào metadata

### 2. Deterministic Chunk IDs

**File: [`rag/vector_store.py`](rag/vector_store.py)**

- Thay đổi ID generation trong `add_documents()`:
  - Loại bỏ UUID random
  - Sử dụng deterministic ID: `f"{file_path}_{chunk_index}"` hoặc hash-based ID
  - Đảm bảo cùng file_path + chunk_index luôn tạo cùng ID

### 3. Implement Upsert Mechanism

**File: [`rag/vector_store.py`](rag/vector_store.py)**

- Thêm method `get_document_version(file_path)` để lấy version metadata của document hiện tại
- Thêm method `upsert_documents()` thay thế cho `add_documents()`:
  - Check xem document đã tồn tại chưa
  - So sánh file_hash với version hiện tại
  - Nếu document changed: delete old chunks, add new chunks
  - Nếu document unchanged: skip hoặc update metadata
- Sử dụng ChromaDB's `update()` method nếu cần update existing chunks

**File: [`ingestion/pipeline.py`](ingestion/pipeline.py)**

- Thay đổi `run_ingestion()` để sử dụng `upsert_documents()` thay vì `add_documents()`
- Trước khi upsert, check và compare document versions

### 4. Backward Compatibility

- Đảm bảo existing chunks không có version metadata vẫn hoạt động
- Khi query existing chunks, treat missing version fields như version 0 hoặc unknown
- Migration: existing chunks sẽ được update khi re-ingested

## Implementation Details

### Chunk ID Strategy

- **Option 1**: `f"{file_path}_{chunk_index}"` - Simple nhưng có thể conflict nếu file_path không unique
- **Option 2**: `f"{file_path_hash}_{chunk_index}"` - Hash file_path để tránh special characters
- **Recommendation**: Option 2 với hash ngắn (first 8-12 chars của SHA256)

### Version Comparison Logic

```python
def should_reingest(current_hash: str, new_hash: str) -> bool:
    return current_hash != new_hash
```

### Upsert Flow

1. Load documents và calculate hashes
2. For each document:

   - Get existing chunks và version metadata
   - Compare hash
   - If changed: delete old chunks, add new chunks với new version
   - If unchanged: skip (hoặc update metadata only)

3. Return statistics: updated_count, skipped_count, new_count

## Files to Modify

1. [`ingestion/metadata_schema.py`](ingestion/metadata_schema.py) - Add version fields
2. [`ingestion/pipeline.py`](ingestion/pipeline.py) - Add hash calculation và version tracking
3. [`rag/vector_store.py`](rag/vector_store.py) - Implement upsert mechanism và deterministic IDs

## Testing Considerations

- Test re-ingestion của same document (should update, not duplicate)
- Test re-ingestion của unchanged document (should skip)
- Test re-ingestion của changed document (should update)
- Test backward compatibility với existing chunks không có version metadata

### To-dos

- [ ] Add FILE_HASH, FILE_MTIME, DOCUMENT_VERSION fields to MetadataFields class and DocumentMetadata model in metadata_schema.py
- [ ] Create calculate_file_hash() and get_file_metadata() helper functions in pipeline.py to compute SHA256 hash and modification time
- [ ] Modify run_ingestion() to calculate and include file hash and mtime in document metadata
- [ ] Change chunk ID generation in VectorStore.add_documents() to use deterministic IDs (file_path_hash + chunk_index) instead of UUID
- [ ] Add get_document_version() method to VectorStore to retrieve version metadata for existing documents
- [ ] Implement upsert_documents() method in VectorStore that checks version, deletes old chunks if changed, and adds/updates new chunks
- [ ] Modify run_ingestion() to use upsert_documents() instead of add_documents()
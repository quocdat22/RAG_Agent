## RAG Agent MVP (Azure OpenAI)

### Chạy nhanh

- **Cài dependency**:

```bash
pip install -r requirements.txt
```

- **Tạo file `.env`** (copy từ ví dụ dưới, chỉnh endpoint/key của bạn):

```bash
cp .env.example .env  # trên Windows có thể copy thủ công
```

- **Chạy API**:

```bash
uvicorn api.main:app --reload
```

- **Ingest thư mục tài liệu**:
  - Gọi API `POST /ingest-folder` với JSON: `{ "folder_path": "path/to/docs" }`
  - Hoặc dùng script:

```bash
python scripts/ingest_sample.py path/to/docs
```

- **Query**:
  - Gọi API `POST /query` với JSON: `{ "query": "câu hỏi" }`
  - Hoặc dùng script:

```bash
python scripts/query_sample.py "câu hỏi nội bộ"
```



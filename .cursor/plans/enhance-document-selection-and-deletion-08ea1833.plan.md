<!-- 08ea1833-beb8-4edc-8ac8-a880ceb5b385 a54d30b5-faaa-4566-ba95-8078a16f8cef -->
## Điều chỉnh quy trình áp dụng tài liệu cho cuộc trò chuyện

### Mục tiêu

- **Bỏ nút "Áp dụng cho cuộc trò chuyện"** trong `sidebar_ingestion`.
- **Mặc định chọn tất cả tài liệu đã ingest** khi người dùng chưa chọn gì.
- **Thêm 1 nút toggle chọn/bỏ chọn tất cả** tài liệu đã ingest.
- **Khi người dùng gửi tin nhắn đầu tiên (hoặc mỗi lần gửi)**, tự động lưu danh sách tài liệu đang được chọn (`selected_documents`) cho conversation đó và retriever chỉ dùng các tài liệu này.

### 1. Cập nhật UI chọn tài liệu (`ui/app.py` – `sidebar_ingestion`)

- Khôi phục/điều chỉnh logic:
- Nếu có `ingested_docs` và `selected_documents` đang rỗng → tự động set `selected_documents = tất_cả_file` (chạy 1 lần mỗi session/hoặc mỗi khi ingest mới).
- Thêm nút **"Chọn/Bỏ chọn tất cả"**:
- Nếu hiện tại **đã chọn ít nhất 1 file** → nút hiển thị "🧹 Bỏ chọn tất cả" và khi bấm sẽ `selected_documents = []`.
- Nếu hiện tại **không có file nào được chọn** và có `ingested_docs` → nút hiển thị "✅ Chọn tất cả" và khi bấm sẽ `selected_documents = tất_cả_file`.
- Xóa toàn bộ code liên quan đến nút **"🔗 Áp dụng cho cuộc trò chuyện"** và các đoạn gọi `update_selected_documents` trong `sidebar_ingestion`.
- (Tùy chọn, đơn giản hóa) Phần hiển thị "🔗 Tài liệu đang dùng cho cuộc trò chuyện" có thể:
- Hoặc bỏ hẳn.
- Hoặc chỉ hiển thị `selected_documents` hiện tại (không phụ thuộc DB), miễn UX của bạn.

### 2. Đồng bộ hóa tài liệu với conversation khi chat (`rag/pipeline.py` và `ui/app.py`)

- Trong `main_chat` (`ui/app.py`):
- Sau khi đảm bảo có `conversation_id` (đã tạo hoặc đang chọn) và **trước khi gọi `answer_query`**, lấy `selected_docs = st.session_state.get("selected_documents", [])`.
- Gọi `store.update_selected_documents(conversation_id, selected_docs or [])` để lưu vào DB.
- Trong `answer_query` (`rag/pipeline.py` – đã có sẵn logic `get_selected_documents` và `allowed_file_paths`):
- Giữ nguyên logic hiện tại: nếu DB trả về danh sách file_paths → dùng để filter; nếu rỗng → không filter (dùng tất cả tài liệu).

### 3. Rà soát lại session state & dọn dư thừa

- `sidebar_conversations`:
- Có thể bỏ hoặc giữ `conversation_documents` nếu chỉ dùng để hiển thị; nhưng **retriever sẽ dựa trên DB + selected_documents cập nhật lúc chat**, không cần apply thủ công nữa.
- Đảm bảo không còn chỗ nào gọi `update_selected_documents` ngoại trừ luồng **bắt đầu chat**.

### 4. Kiểm thử

- Ingest 2 file A, B.
- Case 1: Không thao tác chọn → nút mặc định chọn tất cả, hỏi 1 câu → conversation dùng cả A và B.
- Case 2: Bỏ chọn tất cả rồi chỉ chọn A → gửi câu hỏi → chỉ A được dùng.
- Case 3: Đổi lại chỉ chọn B và tiếp tục chat trong cùng conversation → check retriever chỉ dùng B (do cập nhật lại khi gửi message).

### To-dos

- [ ] Thêm logic mặc định chọn tất cả và nút toggle chọn/bỏ chọn tất cả trong sidebar_ingestion
- [ ] Loại bỏ nút và logic "Áp dụng cho cuộc trò chuyện" trong sidebar_ingestion
- [ ] Trong main_chat, tự động gọi update_selected_documents với selected_documents ngay trước khi answer_query
- [ ] Rà soát/bỏ hoặc đơn giản hóa conversation_documents trong UI để không xung đột với logic mới
- [ ] Test luồng: mặc định tất cả, chỉ A, chỉ B, đổi lựa chọn trong cùng conversation
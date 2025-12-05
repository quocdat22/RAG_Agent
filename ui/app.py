import os
import sys
import tempfile
from pathlib import Path

import streamlit as st

# Ensure project root is on sys.path when running via `streamlit run ui/app.py`
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from ingestion.pipeline import run_ingestion
from rag.exceptions import RAGAgentException
from rag.pipeline import answer_query
from rag.vector_store import get_vector_store
from storage.conversation_store import get_conversation_store
from ui.state_manager import StateManager


st.set_page_config(page_title="RAG Agent MVP", page_icon="💬", layout="wide")


def save_uploaded_files(files) -> str:
    """
    Save uploaded files into a temporary directory and return its path.
    """
    tmp_dir = Path(tempfile.mkdtemp(prefix="rag_uploads_"))
    for f in files:
        out_path = tmp_dir / f.name
        with out_path.open("wb") as out_f:
            out_f.write(f.read())
    return str(tmp_dir)


def sidebar_conversations():
    """Sidebar for conversation management."""
    st.sidebar.header("💬 Cuộc trò chuyện")
    
    store = get_conversation_store()
    
    # Create new conversation button
    if st.sidebar.button("➕ Cuộc trò chuyện mới", use_container_width=True):
        new_conv_id = store.create_conversation()
        StateManager.set_current_conversation_id(new_conv_id)
        StateManager.clear_messages()
        st.rerun()
    
    # Toggle for using conversation history
    use_history = st.sidebar.checkbox(
        "Sử dụng lịch sử cuộc trò chuyện",
        value=StateManager.get_use_history(),
        help="Bật để LLM nhớ các câu hỏi/trả lời trước đó trong cùng cuộc trò chuyện"
    )
    StateManager.set_use_history(use_history)
    
    st.sidebar.divider()
    
    # List conversations
    try:
        conversations = store.list_conversations(limit=20)
        
        if conversations:
            st.sidebar.subheader("Danh sách cuộc trò chuyện")
            
            for conv in conversations:
                conv_id = conv["id"]
                title = conv["title"]
                message_count = conv.get("message_count", 0)
                
                # Create a container for each conversation
                col1, col2 = st.sidebar.columns([4, 1])
                
                with col1:
                    is_selected = st.button(
                        f"💬 {title}",
                        key=f"conv_{conv_id}",
                        use_container_width=True,
                        help=f"{message_count} tin nhắn"
                    )
                    if is_selected:
                        StateManager.set_current_conversation_id(conv_id)
                        # Load messages for this conversation
                        messages = store.get_messages(conv_id)
                        StateManager.set_messages([
                            {"role": msg["role"], "content": msg["content"]}
                            for msg in messages
                        ])
                        st.rerun()
                
                with col2:
                    if st.button("🗑️", key=f"del_{conv_id}", help="Xóa"):
                        store.delete_conversation(conv_id)
                        if StateManager.get_current_conversation_id() == conv_id:
                            StateManager.reset_conversation_state()
                        st.rerun()
        else:
            st.sidebar.info("Chưa có cuộc trò chuyện nào.")
    except Exception as e:
        st.sidebar.error(
            "⚠️ Không thể tải danh sách cuộc trò chuyện. "
            "Vui lòng làm mới trang hoặc thử lại sau."
        )


def sidebar_ingestion():
    st.sidebar.header("📂 Ingestion")
    uploaded_files = st.sidebar.file_uploader(
        "Upload tài liệu (PDF/DOCX/XLSX/HTML/MD)", accept_multiple_files=True
    )

    if st.sidebar.button("Ingest tài liệu") and uploaded_files:
        with st.spinner("Đang lưu file và ingest vào vector store..."):
            try:
                folder = save_uploaded_files(uploaded_files)
                count = run_ingestion(folder)
                st.sidebar.success(f"Ingest xong {count} chunks từ thư mục tạm.")
                # Reset document state after ingestion
                StateManager.reset_document_state()
            except RAGAgentException as e:
                st.sidebar.error(f"⚠️ {e.user_message}")
            except Exception as e:
                st.sidebar.error(
                    "⚠️ Đã xảy ra lỗi khi xử lý tài liệu. "
                    "Vui lòng kiểm tra định dạng file và thử lại."
                )
    
    # Display ingested documents
    st.sidebar.divider()
    st.sidebar.subheader("📄 Tài liệu đã ingest")
    
    try:
        # Cache the document list to avoid querying on every rerun
        ingested_docs = StateManager.get_ingested_docs_cache()
        if ingested_docs is None:
            vs = get_vector_store()
            ingested_docs = vs.get_all_documents()
            StateManager.set_ingested_docs_cache(ingested_docs)

        # Clear view_document if the document no longer exists
        view_doc = StateManager.get_view_document()
        if view_doc:
            doc_keys = [doc.get("file_path") or doc.get("name", "Unknown") for doc in ingested_docs]
            if view_doc not in doc_keys:
                StateManager.set_view_document(None)
        
        # Mặc định: nếu chưa chọn gì thì chọn tất cả tài liệu (chỉ 1 lần)
        selected_docs = StateManager.get_selected_documents()
        auto_done = StateManager.get_auto_select_docs_done()
        if (
            ingested_docs
            and not selected_docs
            and not auto_done
        ):
            StateManager.set_selected_documents([
                (doc.get("file_path") or doc.get("name", "Unknown"))
                for doc in ingested_docs
            ])
            StateManager.set_auto_select_docs_done(True)
        
        if ingested_docs:
            selected_docs = StateManager.get_selected_documents()
            # Bulk actions section
            if selected_docs:
                col1, col2 = st.sidebar.columns(2)
                with col1:
                    if st.button("📋 Xem chi tiết", use_container_width=True, key="view_selected"):
                        StateManager.set_view_document(selected_docs[0])
                        st.rerun()
                with col2:
                    if st.button("🗑️ Xóa đã chọn", use_container_width=True, key="delete_selected"):
                        StateManager.set_show_delete_confirm(True)
                        StateManager.set_delete_targets(selected_docs.copy())
                        st.rerun()
            # Nút toggle chọn/bỏ chọn tất cả
            if ingested_docs:
                if selected_docs:
                    # Đang chọn ít nhất 1 tài liệu -> cho phép bỏ chọn tất cả
                    if st.sidebar.button(
                        "🧹 Bỏ chọn tất cả",
                        use_container_width=True,
                        key="clear_selected_docs",
                    ):
                        StateManager.clear_selected_documents()
                        # Đánh dấu đã tùy chỉnh, không auto-select lại
                        StateManager.set_auto_select_docs_done(True)
                        st.rerun()
                else:
                    # Không có tài liệu nào được chọn -> cho phép chọn tất cả
                    if st.sidebar.button(
                        "✅ Chọn tất cả",
                        use_container_width=True,
                        key="select_all_docs",
                    ):
                        StateManager.set_selected_documents([
                            (doc.get("file_path") or doc.get("name", "Unknown"))
                            for doc in ingested_docs
                        ])
                        StateManager.set_auto_select_docs_done(True)
                        st.rerun()
            
            # Delete confirmation dialog
            if StateManager.get_show_delete_confirm():
                st.sidebar.warning("⚠️ Xác nhận xóa")
                targets = StateManager.get_delete_targets()
                for target in targets:
                    st.sidebar.text(f"• {target}")
                col1, col2 = st.sidebar.columns(2)
                with col1:
                    if st.button("✅ Xác nhận", use_container_width=True, key="confirm_delete"):
                        vs = get_vector_store()
                        deleted_count = 0
                        for file_path in targets:
                            count = vs.delete_document(file_path)
                            deleted_count += count
                        st.sidebar.success(f"Đã xóa {deleted_count} chunks từ {len(targets)} tài liệu.")
                        # Reset document state after deletion
                        StateManager.reset_document_state()
                        st.rerun()
                with col2:
                    if st.button("❌ Hủy", use_container_width=True, key="cancel_delete"):
                        StateManager.clear_delete_confirm()
                        st.rerun()
            
            st.sidebar.divider()
            
            # Document list with checkboxes
            for idx, doc in enumerate(ingested_docs):
                doc_name = doc.get("name", "Unknown")
                chunk_count = doc.get("chunk_count", 0)
                file_path = doc.get("file_path", "")
                
                # Use file_path as the key for selection
                doc_key = file_path or doc_name
                
                # Checkbox for selection - Streamlit automatically handles rerun on change
                is_selected = st.sidebar.checkbox(
                    f"📄 {doc_name}",
                    value=StateManager.is_document_selected(doc_key),
                    key=f"doc_checkbox_{idx}",
                    help=f"{chunk_count} chunks"
                )
                
                # Update selection based on checkbox state
                # This runs after Streamlit reruns due to checkbox change
                if is_selected:
                    StateManager.add_selected_document(doc_key)
                else:
                    StateManager.remove_selected_document(doc_key)
                # Người dùng đã tương tác lựa chọn thủ công
                StateManager.set_auto_select_docs_done(True)
                
                # Action buttons for each document
                col1, col2, col3 = st.sidebar.columns([2, 2, 1])
                
                with col1:
                    if st.button("Chi tiết", key=f"view_{idx}", use_container_width=True):
                        StateManager.set_view_document(doc_key)
                        st.rerun()
                
                with col2:
                    if st.button("🗑️ Xóa", key=f"delete_{idx}", use_container_width=True):
                        StateManager.set_show_delete_confirm(True)
                        StateManager.set_delete_targets([doc_key])
                        st.rerun()
                
                with col3:
                    st.caption(f"{chunk_count}")
                
                # Show file path if different from name
                if file_path and file_path != doc_name:
                    st.sidebar.caption(f"📍 {file_path}")
        else:
            st.sidebar.info("Chưa có tài liệu nào được ingest.")
    except Exception as e:
        st.sidebar.error(
            "⚠️ Không thể tải danh sách tài liệu. "
            "Vui lòng làm mới trang hoặc thử lại sau."
        )
    
    # View document details modal/expander
    view_doc = StateManager.get_view_document()
    if view_doc:
        st.sidebar.divider()
        st.sidebar.subheader("📋 Chi tiết tài liệu")
        
        try:
            vs = get_vector_store()
            chunks = vs.get_document_chunks(view_doc)
            
            if chunks:
                st.sidebar.write(f"**Tổng số chunks:** {len(chunks)}")
                
                # Show chunks in an expander
                for idx, chunk in enumerate(chunks, 1):
                    with st.sidebar.expander(f"Chunk {idx}", expanded=False):
                        st.text_area(
                            "Nội dung",
                            value=chunk.get("content", ""),
                            height=150,
                            key=f"chunk_content_{idx}",
                            disabled=True
                        )
                        metadata = chunk.get("metadata", {})
                        if metadata:
                            st.caption("Metadata:")
                            for key, value in metadata.items():
                                st.caption(f"  • {key}: {value}")
            else:
                st.sidebar.warning("Không tìm thấy chunks cho tài liệu này.")
            
            if st.sidebar.button("✖️ Đóng", key="close_view"):
                StateManager.set_view_document(None)
                st.rerun()
        except Exception as e:
            st.sidebar.error(
                "⚠️ Không thể tải chi tiết tài liệu. "
                "Vui lòng thử lại sau."
            )
            if st.sidebar.button("✖️ Đóng", key="close_view_error"):
                StateManager.set_view_document(None)
                st.rerun()


def main_chat():
    st.title("RAG Agent MVP (Azure OpenAI)")
    st.caption("Chat dựa trên tài liệu nội bộ đã ingest.")

    store = get_conversation_store()
    
    # Display current conversation info
    conversation_id = StateManager.get_current_conversation_id()
    if conversation_id:
        conv = store.get_conversation(conversation_id)
        if conv:
            st.caption(f"📝 {conv['title']}")

    # Display messages
    messages = StateManager.get_messages()
    for msg in messages:
        role = "user" if msg["role"] == "user" else "assistant"
        with st.chat_message(role):
            st.markdown(msg["content"])
            # Display sources if available
            if role == "assistant" and "sources" in msg and msg["sources"]:
                with st.expander("Nguồn tham khảo"):
                    for i, src in enumerate(msg["sources"], start=1):
                        st.write(f"{i}. {src}")

    user_input = st.chat_input("Nhập câu hỏi về tài liệu nội bộ...")
    if user_input:
        # Ensure we have a conversation
        conversation_id = StateManager.get_current_conversation_id()
        if not conversation_id:
            conversation_id = store.create_conversation()
            StateManager.set_current_conversation_id(conversation_id)

        # Tự động áp dụng các tài liệu đang được chọn cho cuộc trò chuyện hiện tại
        selected_docs = StateManager.get_selected_documents()
        # Lưu vào DB để retriever giới hạn theo tài liệu đã chọn
        store.update_selected_documents(conversation_id, selected_docs or [])

        # Add user message to state
        StateManager.append_message({"role": "user", "content": user_input})
        
        # Save user message to DB
        store.add_message(
            conversation_id=conversation_id,
            role="user",
            content=user_input,
        )
        
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            with st.spinner("Đang truy vấn RAG..."):
                try:
                    result = answer_query(
                        user_input,
                        conversation_id=conversation_id,
                        use_history=StateManager.get_use_history(),
                    )
                except RAGAgentException as e:
                    # Use user-friendly message from custom exception
                    error_msg = e.user_message
                    st.error(f"⚠️ {error_msg}")
                    StateManager.append_message(
                        {"role": "assistant", "content": f"⚠️ {error_msg}"}
                    )
                    # Save error message to DB
                    try:
                        store.add_message(
                            conversation_id=conversation_id,
                            role="assistant",
                            content=f"⚠️ {error_msg}",
                        )
                    except Exception:
                        # If we can't save, just log it
                        pass
                    return
                except Exception as e:
                    # Fallback for unexpected errors
                    error_msg = (
                        "Đã xảy ra lỗi không mong đợi khi xử lý câu hỏi của bạn. "
                        "Vui lòng thử lại sau hoặc liên hệ quản trị viên nếu vấn đề tiếp tục."
                    )
                    st.error(f"⚠️ {error_msg}")
                    StateManager.append_message(
                        {"role": "assistant", "content": f"⚠️ {error_msg}"}
                    )
                    # Save error message to DB
                    try:
                        store.add_message(
                            conversation_id=conversation_id,
                            role="assistant",
                            content=f"⚠️ {error_msg}",
                        )
                    except Exception:
                        # If we can't save, just log it
                        pass
                    return

            # Display answer
            if result.answer:
                st.markdown(result.answer)
            else:
                st.warning("Không nhận được câu trả lời từ LLM. Vui lòng kiểm tra logs.")

            # Display sources
            if result.sources:
                with st.expander("Nguồn tham khảo"):
                    for i, src in enumerate(result.sources, start=1):
                        st.write(f"{i}. {src}")

        # Add assistant message to state
        assistant_msg = {
            "role": "assistant",
            "content": result.answer,
            "sources": result.sources,
        }
        StateManager.append_message(assistant_msg)
        
        # Save assistant message to DB
        store.add_message(
            conversation_id=conversation_id,
            role="assistant",
            content=result.answer,
            sources=result.sources,
        )


def main():
    # Initialize all state at the start
    StateManager.initialize()
    
    sidebar_conversations()
    sidebar_ingestion()
    main_chat()


if __name__ == "__main__":
    main()



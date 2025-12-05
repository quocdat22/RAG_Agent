"""
Centralized state management for Streamlit app.

This module provides a single source of truth for all session state,
preventing race conditions and making state flow easy to track.
"""
from typing import List, Optional, Dict, Any
import streamlit as st


class StateManager:
    """
    Centralized state manager for Streamlit session state.
    
    This class provides:
    - Single source of truth for all state
    - Type-safe access to state values
    - Centralized initialization
    - Clear state flow tracking
    """
    
    # State keys as class constants for type safety
    CURRENT_CONVERSATION_ID = "current_conversation_id"
    USE_HISTORY = "use_history"
    MESSAGES = "messages"
    SELECTED_DOCUMENTS = "selected_documents"
    AUTO_SELECT_DOCS_DONE = "auto_select_docs_done"
    VIEW_DOCUMENT = "view_document"
    INGESTED_DOCS_CACHE = "ingested_docs_cache"
    SHOW_DELETE_CONFIRM = "show_delete_confirm"
    DELETE_TARGETS = "delete_targets"
    
    @staticmethod
    def initialize():
        """Initialize all session state variables with default values."""
        defaults = {
            StateManager.CURRENT_CONVERSATION_ID: None,
            StateManager.USE_HISTORY: False,
            StateManager.MESSAGES: [],
            StateManager.SELECTED_DOCUMENTS: [],
            StateManager.AUTO_SELECT_DOCS_DONE: False,
            StateManager.VIEW_DOCUMENT: None,
            StateManager.SHOW_DELETE_CONFIRM: False,
            StateManager.DELETE_TARGETS: [],
        }
        
        for key, default_value in defaults.items():
            if key not in st.session_state:
                st.session_state[key] = default_value
    
    # Conversation state methods
    @staticmethod
    def get_current_conversation_id() -> Optional[str]:
        """Get current conversation ID."""
        return st.session_state.get(StateManager.CURRENT_CONVERSATION_ID)
    
    @staticmethod
    def set_current_conversation_id(conv_id: Optional[str]):
        """Set current conversation ID."""
        st.session_state[StateManager.CURRENT_CONVERSATION_ID] = conv_id
    
    @staticmethod
    def get_use_history() -> bool:
        """Get use history flag."""
        return st.session_state.get(StateManager.USE_HISTORY, False)
    
    @staticmethod
    def set_use_history(value: bool):
        """Set use history flag."""
        st.session_state[StateManager.USE_HISTORY] = value
    
    # Messages state methods
    @staticmethod
    def get_messages() -> List[Dict[str, Any]]:
        """Get messages list."""
        return st.session_state.get(StateManager.MESSAGES, [])
    
    @staticmethod
    def set_messages(messages: List[Dict[str, Any]]):
        """Set messages list."""
        st.session_state[StateManager.MESSAGES] = messages
    
    @staticmethod
    def append_message(message: Dict[str, Any]):
        """Append a message to the messages list."""
        if StateManager.MESSAGES not in st.session_state:
            st.session_state[StateManager.MESSAGES] = []
        st.session_state[StateManager.MESSAGES].append(message)
    
    @staticmethod
    def clear_messages():
        """Clear all messages."""
        st.session_state[StateManager.MESSAGES] = []
    
    # Document selection state methods
    @staticmethod
    def get_selected_documents() -> List[str]:
        """Get selected documents list."""
        return st.session_state.get(StateManager.SELECTED_DOCUMENTS, [])
    
    @staticmethod
    def set_selected_documents(docs: List[str]):
        """Set selected documents list."""
        st.session_state[StateManager.SELECTED_DOCUMENTS] = docs
    
    @staticmethod
    def add_selected_document(doc_key: str):
        """Add a document to selected documents (no duplicates)."""
        if StateManager.SELECTED_DOCUMENTS not in st.session_state:
            st.session_state[StateManager.SELECTED_DOCUMENTS] = []
        if doc_key not in st.session_state[StateManager.SELECTED_DOCUMENTS]:
            st.session_state[StateManager.SELECTED_DOCUMENTS].append(doc_key)
    
    @staticmethod
    def remove_selected_document(doc_key: str):
        """Remove a document from selected documents."""
        if StateManager.SELECTED_DOCUMENTS in st.session_state:
            if doc_key in st.session_state[StateManager.SELECTED_DOCUMENTS]:
                st.session_state[StateManager.SELECTED_DOCUMENTS].remove(doc_key)
    
    @staticmethod
    def clear_selected_documents():
        """Clear all selected documents."""
        st.session_state[StateManager.SELECTED_DOCUMENTS] = []
    
    @staticmethod
    def is_document_selected(doc_key: str) -> bool:
        """Check if a document is selected."""
        return doc_key in st.session_state.get(StateManager.SELECTED_DOCUMENTS, [])
    
    # Auto-select flag methods
    @staticmethod
    def get_auto_select_docs_done() -> bool:
        """Get auto-select docs done flag."""
        return st.session_state.get(StateManager.AUTO_SELECT_DOCS_DONE, False)
    
    @staticmethod
    def set_auto_select_docs_done(value: bool):
        """Set auto-select docs done flag."""
        st.session_state[StateManager.AUTO_SELECT_DOCS_DONE] = value
    
    # View document state methods
    @staticmethod
    def get_view_document() -> Optional[str]:
        """Get view document key."""
        return st.session_state.get(StateManager.VIEW_DOCUMENT)
    
    @staticmethod
    def set_view_document(doc_key: Optional[str]):
        """Set view document key."""
        st.session_state[StateManager.VIEW_DOCUMENT] = doc_key
    
    # Cache state methods
    @staticmethod
    def get_ingested_docs_cache() -> Optional[List[Dict[str, Any]]]:
        """Get ingested docs cache."""
        return st.session_state.get(StateManager.INGESTED_DOCS_CACHE)
    
    @staticmethod
    def set_ingested_docs_cache(docs: List[Dict[str, Any]]):
        """Set ingested docs cache."""
        st.session_state[StateManager.INGESTED_DOCS_CACHE] = docs
    
    @staticmethod
    def clear_ingested_docs_cache():
        """Clear ingested docs cache."""
        if StateManager.INGESTED_DOCS_CACHE in st.session_state:
            del st.session_state[StateManager.INGESTED_DOCS_CACHE]
    
    # Delete confirmation state methods
    @staticmethod
    def get_show_delete_confirm() -> bool:
        """Get show delete confirm flag."""
        return st.session_state.get(StateManager.SHOW_DELETE_CONFIRM, False)
    
    @staticmethod
    def set_show_delete_confirm(value: bool):
        """Set show delete confirm flag."""
        st.session_state[StateManager.SHOW_DELETE_CONFIRM] = value
    
    @staticmethod
    def get_delete_targets() -> List[str]:
        """Get delete targets list."""
        return st.session_state.get(StateManager.DELETE_TARGETS, [])
    
    @staticmethod
    def set_delete_targets(targets: List[str]):
        """Set delete targets list."""
        st.session_state[StateManager.DELETE_TARGETS] = targets
    
    @staticmethod
    def clear_delete_confirm():
        """Clear delete confirmation state."""
        st.session_state[StateManager.SHOW_DELETE_CONFIRM] = False
        st.session_state[StateManager.DELETE_TARGETS] = []
    
    # Utility methods
    @staticmethod
    def reset_conversation_state():
        """Reset conversation-related state when switching conversations."""
        StateManager.set_current_conversation_id(None)
        StateManager.clear_messages()
    
    @staticmethod
    def reset_document_state():
        """Reset document-related state after ingestion or deletion."""
        StateManager.clear_ingested_docs_cache()
        StateManager.clear_selected_documents()
        StateManager.set_auto_select_docs_done(False)
        StateManager.set_view_document(None)
        StateManager.clear_delete_confirm()

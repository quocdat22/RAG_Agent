import json
import sqlite3
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from config.settings import get_settings


class ConversationStore:
    """Manages conversation and message storage in SQLite database."""

    def __init__(self, db_path: Optional[str] = None):
        settings = get_settings()
        self.db_path = db_path or getattr(settings, "conversation_db_path", "./data/conversations.db")
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
            conn.execute("""
                CREATE TABLE IF NOT EXISTS conversations (
                    id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS messages (
                    id TEXT PRIMARY KEY,
                    conversation_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    sources TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS conversation_documents (
                    conversation_id TEXT NOT NULL,
                    file_path TEXT NOT NULL,
                    PRIMARY KEY (conversation_id, file_path),
                    FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_messages_conversation 
                ON messages(conversation_id)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_messages_created 
                ON messages(created_at)
            """)
            conn.commit()
        finally:
            conn.close()

    def get_selected_documents(self, conversation_id: str) -> List[str]:
        """
        Get the list of file_paths selected for a conversation.
        
        Args:
            conversation_id: ID of the conversation (validated for SQL injection prevention)
        """
        # Validate conversation_id
        if not conversation_id or not isinstance(conversation_id, str):
            raise ValueError("conversation_id must be a non-empty string")
        # Sanitize: only allow alphanumeric, hyphens, and underscores (UUID format)
        if not all(c.isalnum() or c in ('-', '_') for c in conversation_id):
            raise ValueError("conversation_id contains invalid characters")
        
        conn = self._get_connection()
        try:
            rows = conn.execute(
                """
                SELECT file_path 
                FROM conversation_documents 
                WHERE conversation_id = ?
                ORDER BY file_path ASC
                """,
                (conversation_id,),
            ).fetchall()
            return [row["file_path"] for row in rows]
        finally:
            conn.close()

    def update_selected_documents(
        self,
        conversation_id: str,
        file_paths: List[str],
    ) -> None:
        """
        Replace the list of selected documents for a conversation.
        
        Args:
            conversation_id: ID of the conversation (validated for SQL injection prevention)
            file_paths: List of file paths (validated)
        """
        # Validate conversation_id
        if not conversation_id or not isinstance(conversation_id, str):
            raise ValueError("conversation_id must be a non-empty string")
        if not all(c.isalnum() or c in ('-', '_') for c in conversation_id):
            raise ValueError("conversation_id contains invalid characters")
        
        # Validate file_paths
        if not isinstance(file_paths, list):
            raise ValueError("file_paths must be a list")
        # Validate each file path (basic validation - prevent null bytes and extremely long paths)
        for fp in file_paths:
            if not isinstance(fp, str):
                raise ValueError("All file paths must be strings")
            if '\x00' in fp:
                raise ValueError("File path contains null byte")
            if len(fp) > 4096:  # Reasonable maximum path length
                raise ValueError("File path too long")
        
        conn = self._get_connection()
        try:
            conn.execute(
                "DELETE FROM conversation_documents WHERE conversation_id = ?",
                (conversation_id,),
            )
            if file_paths:
                conn.executemany(
                    """
                    INSERT INTO conversation_documents (conversation_id, file_path)
                    VALUES (?, ?)
                    """,
                    [(conversation_id, fp) for fp in file_paths],
                )
            conn.commit()
        finally:
            conn.close()

    def create_conversation(self, title: Optional[str] = None) -> str:
        """
        Create a new conversation and return its ID.
        
        Args:
            title: Optional title for the conversation (validated and sanitized)
        """
        conversation_id = str(uuid.uuid4())
        title = title or "Cuộc trò chuyện mới"
        
        # Validate and sanitize title
        if title:
            if not isinstance(title, str):
                raise ValueError("title must be a string")
            if len(title) > 500:  # Reasonable maximum length
                title = title[:500]
            # Remove null bytes
            title = title.replace('\x00', '')
        
        conn = self._get_connection()
        try:
            conn.execute(
                "INSERT INTO conversations (id, title) VALUES (?, ?)",
                (conversation_id, title)
            )
            conn.commit()
            return conversation_id
        finally:
            conn.close()

    def get_conversation(self, conversation_id: str) -> Optional[Dict[str, Any]]:
        """
        Get conversation details by ID.
        
        Args:
            conversation_id: ID of the conversation (validated for SQL injection prevention)
        """
        # Validate conversation_id
        if not conversation_id or not isinstance(conversation_id, str):
            raise ValueError("conversation_id must be a non-empty string")
        if not all(c.isalnum() or c in ('-', '_') for c in conversation_id):
            raise ValueError("conversation_id contains invalid characters")
        
        conn = self._get_connection()
        try:
            row = conn.execute(
                "SELECT * FROM conversations WHERE id = ?",
                (conversation_id,)
            ).fetchone()
            if row:
                return dict(row)
            return None
        finally:
            conn.close()

    def list_conversations(self, limit: int = 50, offset: int = 0) -> List[Dict[str, Any]]:
        """
        List all conversations, ordered by updated_at descending.
        
        Args:
            limit: Maximum number of conversations to return (must be positive, max 1000)
            offset: Number of conversations to skip (must be non-negative)
        """
        # Validate and sanitize inputs to prevent SQL injection
        if not isinstance(limit, int) or limit < 1:
            raise ValueError("limit must be a positive integer")
        if limit > 1000:
            limit = 1000  # Cap at reasonable maximum
        if not isinstance(offset, int) or offset < 0:
            raise ValueError("offset must be a non-negative integer")
        
        conn = self._get_connection()
        try:
            rows = conn.execute(
                """
                SELECT c.*, COUNT(m.id) as message_count
                FROM conversations c
                LEFT JOIN messages m ON c.id = m.conversation_id
                GROUP BY c.id
                ORDER BY c.updated_at DESC
                LIMIT ? OFFSET ?
                """,
                (limit, offset)
            ).fetchall()
            return [dict(row) for row in rows]
        finally:
            conn.close()

    def delete_conversation(self, conversation_id: str) -> bool:
        """
        Delete a conversation and all its messages.
        
        Args:
            conversation_id: ID of the conversation (validated for SQL injection prevention)
        """
        # Validate conversation_id
        if not conversation_id or not isinstance(conversation_id, str):
            raise ValueError("conversation_id must be a non-empty string")
        if not all(c.isalnum() or c in ('-', '_') for c in conversation_id):
            raise ValueError("conversation_id contains invalid characters")
        
        conn = self._get_connection()
        try:
            cursor = conn.execute(
                "DELETE FROM conversations WHERE id = ?",
                (conversation_id,)
            )
            conn.commit()
            return cursor.rowcount > 0
        finally:
            conn.close()

    def update_conversation_title(self, conversation_id: str, title: str):
        """
        Update conversation title.
        
        Args:
            conversation_id: ID of the conversation (validated for SQL injection prevention)
            title: New title (validated and sanitized)
        """
        # Validate conversation_id
        if not conversation_id or not isinstance(conversation_id, str):
            raise ValueError("conversation_id must be a non-empty string")
        if not all(c.isalnum() or c in ('-', '_') for c in conversation_id):
            raise ValueError("conversation_id contains invalid characters")
        
        # Validate and sanitize title
        if not isinstance(title, str):
            raise ValueError("title must be a string")
        if len(title) > 500:
            title = title[:500]
        title = title.replace('\x00', '')  # Remove null bytes
        
        conn = self._get_connection()
        try:
            conn.execute(
                """
                UPDATE conversations 
                SET title = ?, updated_at = CURRENT_TIMESTAMP
                WHERE id = ?
                """,
                (title, conversation_id)
            )
            conn.commit()
        finally:
            conn.close()

    def add_message(
        self,
        conversation_id: str,
        role: str,
        content: str,
        sources: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        """
        Add a message to a conversation and return message ID.
        
        Args:
            conversation_id: ID of the conversation (validated for SQL injection prevention)
            role: Message role (must be 'user' or 'assistant')
            content: Message content (validated and sanitized)
            sources: Optional list of source dictionaries
        """
        # Validate conversation_id
        if not conversation_id or not isinstance(conversation_id, str):
            raise ValueError("conversation_id must be a non-empty string")
        if not all(c.isalnum() or c in ('-', '_') for c in conversation_id):
            raise ValueError("conversation_id contains invalid characters")
        
        # Validate role
        if role not in ('user', 'assistant'):
            raise ValueError("role must be 'user' or 'assistant'")
        
        # Validate and sanitize content
        if not isinstance(content, str):
            raise ValueError("content must be a string")
        if len(content) > 1000000:  # Reasonable maximum (1MB)
            raise ValueError("content too long")
        content = content.replace('\x00', '')  # Remove null bytes
        
        message_id = str(uuid.uuid4())
        sources_json = json.dumps(sources) if sources else None

        conn = self._get_connection()
        try:
            # Add message
            conn.execute(
                """
                INSERT INTO messages (id, conversation_id, role, content, sources)
                VALUES (?, ?, ?, ?, ?)
                """,
                (message_id, conversation_id, role, content, sources_json)
            )
            # Update conversation updated_at
            conn.execute(
                """
                UPDATE conversations 
                SET updated_at = CURRENT_TIMESTAMP
                WHERE id = ?
                """,
                (conversation_id,)
            )
            # Auto-update title from first user message if title is default
            if role == "user":
                conv = conn.execute(
                    "SELECT title FROM conversations WHERE id = ?",
                    (conversation_id,)
                ).fetchone()
                if conv and conv["title"] == "Cuộc trò chuyện mới":
                    title = content[:50] + "..." if len(content) > 50 else content
                    conn.execute(
                        "UPDATE conversations SET title = ? WHERE id = ?",
                        (title, conversation_id)
                    )
            conn.commit()
            return message_id
        finally:
            conn.close()

    def get_messages(
        self,
        conversation_id: str,
        limit: Optional[int] = None,
        offset: int = 0,
    ) -> List[Dict[str, Any]]:
        """
        Get messages for a conversation, ordered by created_at ascending.
        
        Args:
            conversation_id: ID of the conversation
            limit: Maximum number of messages to return (must be positive, max 1000)
            offset: Number of messages to skip (must be non-negative)
        """
        # Validate conversation_id (basic validation)
        if not conversation_id or not isinstance(conversation_id, str):
            raise ValueError("conversation_id must be a non-empty string")
        # Sanitize: only allow alphanumeric, hyphens, and underscores (UUID format)
        if not all(c.isalnum() or c in ('-', '_') for c in conversation_id):
            raise ValueError("conversation_id contains invalid characters")
        
        # Validate and sanitize limit and offset
        if limit is not None:
            if not isinstance(limit, int) or limit < 1:
                raise ValueError("limit must be a positive integer")
            if limit > 1000:
                limit = 1000  # Cap at reasonable maximum
        if not isinstance(offset, int) or offset < 0:
            raise ValueError("offset must be a non-negative integer")
        
        conn = self._get_connection()
        try:
            query = """
                SELECT * FROM messages
                WHERE conversation_id = ?
                ORDER BY created_at ASC
            """
            params = [conversation_id]
            if limit:
                query += " LIMIT ? OFFSET ?"
                params.extend([limit, offset])
            
            rows = conn.execute(query, params).fetchall()
            messages = []
            for row in rows:
                msg = dict(row)
                if msg["sources"]:
                    try:
                        msg["sources"] = json.loads(msg["sources"])
                    except json.JSONDecodeError:
                        msg["sources"] = []
                else:
                    msg["sources"] = []
                messages.append(msg)
            return messages
        finally:
            conn.close()

    def get_conversation_history(
        self,
        conversation_id: str,
        max_messages: int = 10,
    ) -> List[Dict[str, str]]:
        """
        Get recent conversation history in format for LLM.
        
        Args:
            conversation_id: ID of the conversation (validated)
            max_messages: Maximum number of messages to return (validated)
        """
        # Validate max_messages
        if not isinstance(max_messages, int) or max_messages < 1:
            raise ValueError("max_messages must be a positive integer")
        if max_messages > 100:
            max_messages = 100  # Cap at reasonable maximum
        
        messages = self.get_messages(conversation_id, limit=max_messages)
        # Return only role and content for LLM
        return [
            {"role": msg["role"], "content": msg["content"]}
            for msg in messages
        ]


# Singleton instance
_store_instance: Optional[ConversationStore] = None


def get_conversation_store() -> ConversationStore:
    """Get or create the singleton ConversationStore instance."""
    global _store_instance
    if _store_instance is None:
        _store_instance = ConversationStore()
    return _store_instance


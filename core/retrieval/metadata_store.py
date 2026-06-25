"""
SQLite repository layer — documents, extracted_tables, sessions, messages.
"""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Optional

from models.schemas import ChatMessage, Document, ExtractedTable, SourceRef


def _connect(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def _run_migrations(conn: sqlite3.Connection, migrations_dir: Path) -> None:
    migration_path = migrations_dir / "001_initial.sql"
    sql = migration_path.read_text()
    conn.executescript(sql)
    conn.commit()


class MetadataStore:
    def __init__(self, db_path: Path, migrations_dir: Path):
        self._conn = _connect(db_path)
        _run_migrations(self._conn, migrations_dir)

    # ─── Documents ───────────────────────────────────────────────────────────

    def upsert_document(self, doc: Document) -> None:
        self._conn.execute(
            """
            INSERT INTO documents (id, name, file_type, file_path, page_count,
                chunk_count, status, uploaded_at, error_msg)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                status      = excluded.status,
                page_count  = excluded.page_count,
                chunk_count = excluded.chunk_count,
                error_msg   = excluded.error_msg
            """,
            (
                doc.id, doc.name, doc.file_type, doc.file_path,
                doc.page_count, doc.chunk_count, doc.status,
                doc.uploaded_at.isoformat(), doc.error_msg,
            ),
        )
        self._conn.commit()

    def get_document(self, doc_id: str) -> Optional[Document]:
        row = self._conn.execute(
            "SELECT * FROM documents WHERE id = ?", (doc_id,)
        ).fetchone()
        return _row_to_document(row) if row else None

    def list_documents(self) -> list[Document]:
        rows = self._conn.execute(
            "SELECT * FROM documents ORDER BY uploaded_at DESC"
        ).fetchall()
        return [_row_to_document(r) for r in rows]

    def delete_document(self, doc_id: str) -> None:
        self._conn.execute("DELETE FROM documents WHERE id = ?", (doc_id,))
        self._conn.commit()

    def set_document_status(
        self, doc_id: str, status: str, error_msg: Optional[str] = None
    ) -> None:
        self._conn.execute(
            "UPDATE documents SET status = ?, error_msg = ? WHERE id = ?",
            (status, error_msg, doc_id),
        )
        self._conn.commit()

    # ─── Extracted Tables ─────────────────────────────────────────────────────

    def insert_table(self, table: ExtractedTable) -> None:
        self._conn.execute(
            """
            INSERT INTO extracted_tables
                (id, doc_id, page_number, table_index, header, markdown,
                 json_data, row_count, col_count)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                table.id, table.doc_id, table.page_number, table.table_index,
                table.header, table.markdown, table.json_data,
                table.row_count, table.col_count,
            ),
        )
        self._conn.commit()

    def get_table(self, table_id: str) -> Optional[ExtractedTable]:
        row = self._conn.execute(
            "SELECT * FROM extracted_tables WHERE id = ?", (table_id,)
        ).fetchone()
        return _row_to_table(row) if row else None

    def get_tables_for_doc(self, doc_id: str) -> list[ExtractedTable]:
        rows = self._conn.execute(
            "SELECT * FROM extracted_tables WHERE doc_id = ? ORDER BY page_number, table_index",
            (doc_id,),
        ).fetchall()
        return [_row_to_table(r) for r in rows]

    def get_tables_by_ids(self, table_ids: list[str]) -> list[ExtractedTable]:
        if not table_ids:
            return []
        placeholders = ",".join("?" * len(table_ids))
        rows = self._conn.execute(
            f"SELECT * FROM extracted_tables WHERE id IN ({placeholders})",
            table_ids,
        ).fetchall()
        return [_row_to_table(r) for r in rows]

    def delete_tables_for_doc(self, doc_id: str) -> None:
        self._conn.execute(
            "DELETE FROM extracted_tables WHERE doc_id = ?", (doc_id,)
        )
        self._conn.commit()

    # ─── Sessions & Messages ─────────────────────────────────────────────────

    def create_session(self, session_id: str) -> None:
        self._conn.execute(
            "INSERT OR IGNORE INTO chat_sessions (id, started_at) VALUES (?, ?)",
            (session_id, datetime.utcnow().isoformat()),
        )
        self._conn.commit()

    def save_message(self, msg: ChatMessage) -> None:
        self._conn.execute(
            """
            INSERT INTO messages
                (id, session_id, role, content, sources, generated_code, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                msg.id, msg.session_id, msg.role, msg.content,
                json.dumps([s.model_dump() for s in msg.sources]),
                msg.generated_code,
                msg.created_at.isoformat(),
            ),
        )
        self._conn.commit()

    def get_session_messages(self, session_id: str) -> list[ChatMessage]:
        rows = self._conn.execute(
            "SELECT * FROM messages WHERE session_id = ? ORDER BY created_at",
            (session_id,),
        ).fetchall()
        return [_row_to_message(r) for r in rows]


# ─── Row converters ───────────────────────────────────────────────────────────

def _row_to_document(row: sqlite3.Row) -> Document:
    return Document(
        id=row["id"],
        name=row["name"],
        file_type=row["file_type"],
        file_path=row["file_path"],
        page_count=row["page_count"],
        chunk_count=row["chunk_count"],
        status=row["status"],
        uploaded_at=datetime.fromisoformat(row["uploaded_at"]),
        error_msg=row["error_msg"],
    )


def _row_to_table(row: sqlite3.Row) -> ExtractedTable:
    return ExtractedTable(
        id=row["id"],
        doc_id=row["doc_id"],
        page_number=row["page_number"],
        table_index=row["table_index"],
        header=row["header"] or "",
        markdown=row["markdown"],
        json_data=row["json_data"],
        row_count=row["row_count"],
        col_count=row["col_count"],
    )


def _row_to_message(row: sqlite3.Row) -> ChatMessage:
    sources_raw = json.loads(row["sources"] or "[]")
    sources = [SourceRef(**s) for s in sources_raw]
    return ChatMessage(
        id=row["id"],
        session_id=row["session_id"],
        role=row["role"],
        content=row["content"],
        sources=sources,
        generated_code=row["generated_code"],
        created_at=datetime.fromisoformat(row["created_at"]),
    )

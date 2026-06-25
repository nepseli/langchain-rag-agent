from __future__ import annotations

import uuid
from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


def _uuid() -> str:
    return str(uuid.uuid4())


class Document(BaseModel):
    id: str = Field(default_factory=_uuid)
    name: str
    file_type: str          # pdf | docx | xlsx | csv
    file_path: str
    page_count: Optional[int] = None
    chunk_count: Optional[int] = None
    status: str = "pending"  # pending | indexed | error
    uploaded_at: datetime = Field(default_factory=datetime.utcnow)
    error_msg: Optional[str] = None


class ExtractedTable(BaseModel):
    id: str = Field(default_factory=_uuid)
    doc_id: str
    page_number: Optional[int] = None   # None for Excel/CSV
    table_index: int = 0                # nth table on the page / sheet
    header: str = ""                    # surrounding heading for context
    markdown: str                       # GitHub-flavored markdown table
    json_data: str                      # pd.DataFrame.to_json(orient='records')
    row_count: int
    col_count: int


class Chunk(BaseModel):
    id: str = Field(default_factory=_uuid)
    doc_id: str
    doc_name: str
    page_number: Optional[int] = None
    chunk_type: str                     # text | table
    text: str                           # text content or markdown table
    section_heading: str = ""
    table_id: Optional[str] = None     # FK to ExtractedTable if chunk_type == table


class SourceRef(BaseModel):
    doc_name: str
    page_number: Optional[int] = None
    chunk_type: str
    excerpt: str                        # first 200 chars of the chunk


class ChatMessage(BaseModel):
    id: str = Field(default_factory=_uuid)
    session_id: str
    role: str                           # user | assistant
    content: str
    sources: list[SourceRef] = Field(default_factory=list)
    generated_code: Optional[str] = None   # pandas code used in calculation
    created_at: datetime = Field(default_factory=datetime.utcnow)

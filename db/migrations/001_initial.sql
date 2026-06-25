-- Document registry
CREATE TABLE IF NOT EXISTS documents (
    id          TEXT PRIMARY KEY,
    name        TEXT NOT NULL,
    file_type   TEXT NOT NULL,          -- pdf | docx | xlsx | csv
    file_path   TEXT NOT NULL,
    page_count  INTEGER,
    chunk_count INTEGER,
    status      TEXT DEFAULT 'pending', -- pending | indexed | error
    uploaded_at TEXT NOT NULL,
    error_msg   TEXT
);

-- Extracted tables (dual-representation storage)
CREATE TABLE IF NOT EXISTS extracted_tables (
    id          TEXT PRIMARY KEY,
    doc_id      TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    page_number INTEGER,
    table_index INTEGER NOT NULL DEFAULT 0,
    header      TEXT DEFAULT '',
    markdown    TEXT NOT NULL,
    json_data   TEXT NOT NULL,          -- pd.DataFrame.to_json(orient='records')
    row_count   INTEGER NOT NULL,
    col_count   INTEGER NOT NULL
);

-- Chat sessions
CREATE TABLE IF NOT EXISTS chat_sessions (
    id          TEXT PRIMARY KEY,
    started_at  TEXT NOT NULL
);

-- Conversation messages
CREATE TABLE IF NOT EXISTS messages (
    id              TEXT PRIMARY KEY,
    session_id      TEXT NOT NULL REFERENCES chat_sessions(id) ON DELETE CASCADE,
    role            TEXT NOT NULL,      -- user | assistant
    content         TEXT NOT NULL,
    sources         TEXT DEFAULT '[]',  -- JSON array of SourceRef
    generated_code  TEXT,               -- pandas code if calculation query
    created_at      TEXT NOT NULL
);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_tables_doc_id   ON extracted_tables(doc_id);
CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id);

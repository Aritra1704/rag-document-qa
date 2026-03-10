CREATE SCHEMA IF NOT EXISTS rag_document_qa;
SET search_path TO rag_document_qa, public;

CREATE TABLE IF NOT EXISTS documents (
    id BIGSERIAL PRIMARY KEY,
    folder_path TEXT NOT NULL,
    file_name TEXT NOT NULL,
    file_path TEXT NOT NULL UNIQUE,
    pages_total INTEGER NOT NULL DEFAULT 0,
    pages_processed INTEGER NOT NULL DEFAULT 0,
    status TEXT NOT NULL DEFAULT 'pending',
    error_message TEXT,
    last_modified DOUBLE PRECISION,
    file_size BIGINT,
    last_processed_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS pages (
    id BIGSERIAL PRIMARY KEY,
    document_id BIGINT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    page_number INTEGER NOT NULL,
    status TEXT NOT NULL,
    extraction_method TEXT,
    raw_text TEXT,
    parsed_record_count INTEGER NOT NULL DEFAULT 0,
    error_message TEXT,
    processed_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE(document_id, page_number)
);

CREATE TABLE IF NOT EXISTS parsed_records (
    id BIGSERIAL PRIMARY KEY,
    document_id BIGINT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    page_id BIGINT NOT NULL REFERENCES pages(id) ON DELETE CASCADE,
    serial_number TEXT,
    elector_id TEXT,
    name TEXT,
    relative_name TEXT,
    relative_type TEXT,
    house_number TEXT,
    age INTEGER,
    gender TEXT,
    constituency TEXT,
    section_name TEXT,
    file_name TEXT NOT NULL,
    file_path TEXT NOT NULL,
    page_number INTEGER NOT NULL,
    extraction_method TEXT NOT NULL,
    raw_record_text TEXT NOT NULL,
    name_normalized TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

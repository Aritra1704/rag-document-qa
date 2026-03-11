CREATE SCHEMA IF NOT EXISTS rag_document_qa;
SET search_path TO rag_document_qa, public;

CREATE INDEX IF NOT EXISTS idx_documents_folder_path ON documents(folder_path);
CREATE INDEX IF NOT EXISTS idx_documents_status ON documents(status);
CREATE INDEX IF NOT EXISTS idx_documents_file_path ON documents(file_path);

CREATE INDEX IF NOT EXISTS idx_pages_document_id ON pages(document_id);
CREATE INDEX IF NOT EXISTS idx_pages_status ON pages(status);
CREATE INDEX IF NOT EXISTS idx_pages_document_page ON pages(document_id, page_number);

CREATE INDEX IF NOT EXISTS idx_records_name_normalized ON parsed_records(name_normalized);
CREATE INDEX IF NOT EXISTS idx_records_lower_name ON parsed_records(LOWER(name));
CREATE INDEX IF NOT EXISTS idx_records_elector_id ON parsed_records(elector_id);
CREATE INDEX IF NOT EXISTS idx_records_page_id ON parsed_records(page_id);
CREATE INDEX IF NOT EXISTS idx_records_document_page ON parsed_records(document_id, page_number);
CREATE INDEX IF NOT EXISTS idx_records_document_page_id ON parsed_records(document_id, page_id);

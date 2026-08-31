\set ON_ERROR_STOP on

BEGIN;

CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pgcrypto;

CREATE SCHEMA IF NOT EXISTS pka;

COMMENT ON SCHEMA pka IS
    'Personal Knowledge Assistant application schema.';

CREATE TABLE IF NOT EXISTS pka.database_health_check (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    message TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS pka.documents (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),

    tenant_id VARCHAR(128) NOT NULL DEFAULT 'local',
    document_id TEXT NOT NULL,

    source_path TEXT NOT NULL,
    file_name TEXT NOT NULL,
    document_type VARCHAR(32) NOT NULL,
    title TEXT,

    status VARCHAR(32) NOT NULL DEFAULT 'ready',

    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT documents_tenant_not_blank
        CHECK (btrim(tenant_id) <> ''),

    CONSTRAINT documents_document_id_not_blank
        CHECK (btrim(document_id) <> ''),

    CONSTRAINT documents_file_name_not_blank
        CHECK (btrim(file_name) <> ''),

    CONSTRAINT documents_document_type_valid
        CHECK (
            document_type IN (
                'markdown',
                'text',
                'pdf'
            )
        ),

    CONSTRAINT documents_status_valid
        CHECK (
            status IN (
                'processing',
                'ready',
                'failed',
                'deleted'
            )
        ),

    CONSTRAINT documents_tenant_document_unique
        UNIQUE (tenant_id, document_id)
);

CREATE TABLE IF NOT EXISTS pka.document_versions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),

    document_key UUID NOT NULL
        REFERENCES pka.documents(id)
        ON DELETE CASCADE,

    content_hash CHAR(64) NOT NULL,
    active BOOLEAN NOT NULL DEFAULT TRUE,

    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,

    modified_at TIMESTAMPTZ,
    imported_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    deactivated_at TIMESTAMPTZ,

    CONSTRAINT document_versions_hash_format
        CHECK (content_hash ~ '^[0-9a-f]{64}$'),

    CONSTRAINT document_versions_document_hash_unique
        UNIQUE (document_key, content_hash)
);

CREATE UNIQUE INDEX IF NOT EXISTS
    uq_document_versions_one_active
ON pka.document_versions (document_key)
WHERE active = TRUE;

CREATE TABLE IF NOT EXISTS pka.document_chunks (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),

    tenant_id VARCHAR(128) NOT NULL DEFAULT 'local',

    document_key UUID NOT NULL
        REFERENCES pka.documents(id)
        ON DELETE CASCADE,

    document_version_id UUID NOT NULL
        REFERENCES pka.document_versions(id)
        ON DELETE CASCADE,

    document_id TEXT NOT NULL,
    chunk_id CHAR(64) NOT NULL,
    chunk_index INTEGER NOT NULL,

    content_hash CHAR(64) NOT NULL,
    text_content TEXT NOT NULL,

    source_path TEXT NOT NULL,
    file_name TEXT NOT NULL,
    document_type VARCHAR(32) NOT NULL,

    page_number INTEGER,
    section_path JSONB NOT NULL DEFAULT '[]'::jsonb,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,

    embedding vector(1024) NOT NULL,

    active BOOLEAN NOT NULL DEFAULT TRUE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    deactivated_at TIMESTAMPTZ,

    CONSTRAINT document_chunks_tenant_not_blank
        CHECK (btrim(tenant_id) <> ''),

    CONSTRAINT document_chunks_document_id_not_blank
        CHECK (btrim(document_id) <> ''),

    CONSTRAINT document_chunks_chunk_id_format
        CHECK (chunk_id ~ '^[0-9a-f]{64}$'),

    CONSTRAINT document_chunks_content_hash_format
        CHECK (content_hash ~ '^[0-9a-f]{64}$'),

    CONSTRAINT document_chunks_index_non_negative
        CHECK (chunk_index >= 0),

    CONSTRAINT document_chunks_text_not_blank
        CHECK (btrim(text_content) <> ''),

    CONSTRAINT document_chunks_page_positive
        CHECK (
            page_number IS NULL
            OR page_number >= 1
        ),

    CONSTRAINT document_chunks_document_type_valid
        CHECK (
            document_type IN (
                'markdown',
                'text',
                'pdf'
            )
        ),

    CONSTRAINT document_chunks_tenant_chunk_unique
        UNIQUE (tenant_id, chunk_id)
);

CREATE TABLE IF NOT EXISTS pka.user_memories (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),

    tenant_id VARCHAR(128) NOT NULL,
    user_id VARCHAR(128) NOT NULL,

    memory_type VARCHAR(32) NOT NULL,
    content TEXT NOT NULL,

    source_type VARCHAR(32) NOT NULL,
    source_reference TEXT,

    confidence DOUBLE PRECISION NOT NULL,
    status VARCHAR(32) NOT NULL DEFAULT 'candidate',

    embedding vector(1024),

    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    last_confirmed_at TIMESTAMPTZ,
    expires_at TIMESTAMPTZ,
    deleted_at TIMESTAMPTZ,

    CONSTRAINT user_memories_tenant_not_blank
        CHECK (btrim(tenant_id) <> ''),

    CONSTRAINT user_memories_user_not_blank
        CHECK (btrim(user_id) <> ''),

    CONSTRAINT user_memories_content_not_blank
        CHECK (btrim(content) <> ''),

    CONSTRAINT user_memories_type_valid
        CHECK (
            memory_type IN (
                'preference',
                'profile',
                'instruction'
            )
        ),

    CONSTRAINT user_memories_source_type_valid
        CHECK (
            source_type IN (
                'explicit_user_input',
                'agent_candidate',
                'manual_admin'
            )
        ),

    CONSTRAINT user_memories_confidence_valid
        CHECK (
            confidence >= 0.0
            AND confidence <= 1.0
        ),

    CONSTRAINT user_memories_status_valid
        CHECK (
            status IN (
                'candidate',
                'active',
                'rejected',
                'deleted'
            )
        )
);

CREATE INDEX IF NOT EXISTS idx_documents_tenant_status
ON pka.documents (tenant_id, status);

CREATE INDEX IF NOT EXISTS idx_document_versions_document
ON pka.document_versions (document_key, active);

CREATE INDEX IF NOT EXISTS idx_document_chunks_document
ON pka.document_chunks (
    tenant_id,
    document_id,
    active
);

CREATE INDEX IF NOT EXISTS idx_document_chunks_version
ON pka.document_chunks (
    document_version_id,
    active
);

CREATE INDEX IF NOT EXISTS idx_document_chunks_file_name
ON pka.document_chunks (
    tenant_id,
    file_name
);

CREATE INDEX IF NOT EXISTS idx_document_chunks_content_hash
ON pka.document_chunks (
    tenant_id,
    content_hash
);

CREATE INDEX IF NOT EXISTS idx_document_chunks_metadata
ON pka.document_chunks
USING GIN (metadata);

CREATE INDEX IF NOT EXISTS idx_document_chunks_embedding_hnsw
ON pka.document_chunks
USING hnsw (embedding vector_cosine_ops)
WHERE active = TRUE;

CREATE INDEX IF NOT EXISTS idx_user_memories_owner_status
ON pka.user_memories (
    tenant_id,
    user_id,
    status
);

CREATE INDEX IF NOT EXISTS idx_user_memories_expires
ON pka.user_memories (expires_at)
WHERE status = 'active';

CREATE INDEX IF NOT EXISTS idx_user_memories_embedding_hnsw
ON pka.user_memories
USING hnsw (embedding vector_cosine_ops)
WHERE embedding IS NOT NULL
  AND status = 'active';

COMMIT;
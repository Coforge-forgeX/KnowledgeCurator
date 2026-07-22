-- Migration: Create indexing_jobs table for state tracking
-- Purpose: Track indexing job state for retry/resume functionality

CREATE TABLE IF NOT EXISTS indexing_jobs (
    -- Primary key
    job_id VARCHAR(255) PRIMARY KEY,

    -- Job details
    workspace_id INTEGER NOT NULL,
    document_url TEXT NOT NULL,
    kb_id INTEGER,

    -- State tracking
    state VARCHAR(50) NOT NULL DEFAULT 'pending',
    checkpoint_data JSONB DEFAULT '{}',

    -- Retry tracking
    retry_count INTEGER DEFAULT 0,
    last_error TEXT,

    -- Timestamps
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    started_at TIMESTAMP WITH TIME ZONE,
    completed_at TIMESTAMP WITH TIME ZONE,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,

    -- Indexes for common queries
    INDEX idx_indexing_jobs_state (state),
    INDEX idx_indexing_jobs_workspace (workspace_id),
    INDEX idx_indexing_jobs_created (created_at),
    INDEX idx_indexing_jobs_updated (updated_at)
);

-- Add comments
COMMENT ON TABLE indexing_jobs IS 'Tracks indexing job state for retry and resume functionality';
COMMENT ON COLUMN indexing_jobs.job_id IS 'Unique job identifier from queue message';
COMMENT ON COLUMN indexing_jobs.state IS 'Current state: pending, downloading, downloaded, extracting, extracted, indexing, indexed, updating_metadata, completed, failed, retrying';
COMMENT ON COLUMN indexing_jobs.checkpoint_data IS 'JSON checkpoint data for resume: file info, extracted text path, doc_id, etc';
COMMENT ON COLUMN indexing_jobs.retry_count IS 'Number of retry attempts';

-- Create function to auto-update updated_at
CREATE OR REPLACE FUNCTION update_indexing_jobs_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Create trigger
CREATE TRIGGER trigger_update_indexing_jobs_updated_at
    BEFORE UPDATE ON indexing_jobs
    FOR EACH ROW
    EXECUTE FUNCTION update_indexing_jobs_updated_at();

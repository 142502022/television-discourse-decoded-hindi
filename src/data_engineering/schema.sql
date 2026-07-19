CREATE TABLE IF NOT EXISTS episodes (
    episode_id VARCHAR(128) PRIMARY KEY,
    source VARCHAR(255) NOT NULL,
    air_date DATE,
    duration DOUBLE PRECISION,
    processed_at TIMESTAMP NOT NULL
);

CREATE TABLE IF NOT EXISTS segments (
    segment_id VARCHAR(128) PRIMARY KEY,
    episode_id VARCHAR(128) NOT NULL REFERENCES episodes(episode_id),
    speaker_id VARCHAR(128) NOT NULL,
    start_time DOUBLE PRECISION NOT NULL,
    end_time DOUBLE PRECISION NOT NULL
);

CREATE INDEX IF NOT EXISTS ix_segments_episode_id ON segments(episode_id);

CREATE TABLE IF NOT EXISTS transcripts (
    segment_id VARCHAR(128) PRIMARY KEY REFERENCES segments(segment_id),
    transcript_text TEXT NOT NULL,
    language VARCHAR(16) NOT NULL,
    whisper_confidence DOUBLE PRECISION
);

CREATE TABLE IF NOT EXISTS toxicity_scores (
    segment_id VARCHAR(128) PRIMARY KEY REFERENCES segments(segment_id),
    toxicity_score DOUBLE PRECISION NOT NULL,
    detoxify_labels JSONB NOT NULL
);

CREATE TABLE IF NOT EXISTS gender_age_labels (
    segment_id VARCHAR(128) PRIMARY KEY REFERENCES segments(segment_id),
    speaker_gender VARCHAR(32) NOT NULL,
    age_estimate INTEGER,
    confidence DOUBLE PRECISION
);


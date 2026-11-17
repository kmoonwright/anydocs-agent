-- Synthetic usage metrics about the documentation itself.
--
-- The Analyst (build step 8) answers numeric questions from here. The data is seeded
-- deterministically so numeric evals have exact expected values: a question like "which section
-- had the most search misses last month" has one right answer, not an approximately right one.

CREATE TABLE IF NOT EXISTS page_views (
    day           TEXT NOT NULL,          -- YYYY-MM-DD
    corpus        TEXT NOT NULL,
    doc_id        TEXT NOT NULL,
    section       TEXT NOT NULL,          -- top-level heading, e.g. 'Retries'
    views         INTEGER NOT NULL,
    unique_users  INTEGER NOT NULL,
    PRIMARY KEY (day, corpus, doc_id)
);

CREATE TABLE IF NOT EXISTS search_misses (
    day           TEXT NOT NULL,
    corpus        TEXT NOT NULL,
    query         TEXT NOT NULL,
    section       TEXT NOT NULL,          -- the section the query was aiming at
    misses        INTEGER NOT NULL,       -- searches that returned nothing usable
    PRIMARY KEY (day, corpus, query)
);

CREATE TABLE IF NOT EXISTS answer_latency (
    day            TEXT NOT NULL,
    corpus         TEXT NOT NULL,
    section        TEXT NOT NULL,
    answers        INTEGER NOT NULL,
    p50_ms         INTEGER NOT NULL,
    p95_ms         INTEGER NOT NULL,
    PRIMARY KEY (day, corpus, section)
);

CREATE TABLE IF NOT EXISTS citations (
    day           TEXT NOT NULL,
    corpus        TEXT NOT NULL,
    doc_id        TEXT NOT NULL,
    citations     INTEGER NOT NULL,       -- times this doc was cited in an answer
    PRIMARY KEY (day, corpus, doc_id)
);

CREATE INDEX IF NOT EXISTS page_views_section ON page_views (corpus, section, day);
CREATE INDEX IF NOT EXISTS search_misses_section ON search_misses (corpus, section, day);

PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS deals(
  deal_id TEXT PRIMARY KEY,
  company TEXT,
  owner TEXT,
  stage TEXT,
  amount INTEGER,
  labels_json TEXT DEFAULT '[]',
  start_date TEXT,
  credit_json TEXT DEFAULT '{}',
  stakeholders_json TEXT DEFAULT '[]',
  schedule_json TEXT DEFAULT '[]',   -- ★ここが必要
  progress_pct INTEGER DEFAULT 0,
  is_lost INTEGER DEFAULT 0,
  last_stage_before_lost TEXT,
  created_at TEXT DEFAULT (datetime('now')),
  updated_at TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS meetings(
  meeting_id TEXT PRIMARY KEY,
  company TEXT NOT NULL,
  meeting_date TEXT NOT NULL,
  owner_bank TEXT NOT NULL,
  owner_client TEXT NOT NULL,
  prev_stage TEXT NOT NULL,
  amount INTEGER,
  meetings_count INTEGER DEFAULT 1,
  transcript_raw TEXT,
  diarized_json TEXT,
  selected_stage TEXT,
  labels_json TEXT DEFAULT '[]',
  created_at TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS transcript_turns(
  id TEXT PRIMARY KEY,
  meeting_id TEXT NOT NULL,
  turn_index INTEGER,
  speaker_label TEXT,
  speaker_name TEXT,
  start_sec REAL,
  end_sec REAL,
  text TEXT
);

CREATE TABLE IF NOT EXISTS minutes(
  id TEXT PRIMARY KEY,
  meeting_id TEXT NOT NULL,
  version INTEGER,
  is_current INTEGER,
  minutes_md TEXT,
  manager_report TEXT,
  next_actions_json TEXT,
  docx_path TEXT,
  pdf_path TEXT,
  created_at TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS risks(
  id TEXT PRIMARY KEY,
  meeting_id TEXT NOT NULL,
  category TEXT,
  severity TEXT,
  evidence_span TEXT,
  context_summary TEXT,
  at TEXT,
  action TEXT
);

CREATE TABLE IF NOT EXISTS stage_evidence(
  id TEXT PRIMARY KEY,
  meeting_id TEXT NOT NULL,
  speaker_name TEXT,
  quote TEXT,
  at TEXT
);

CREATE TABLE IF NOT EXISTS chat_edits(
  id TEXT PRIMARY KEY,
  meeting_id TEXT NOT NULL,
  user_text TEXT,
  applied_version INTEGER,
  created_at TEXT DEFAULT (datetime('now'))
);

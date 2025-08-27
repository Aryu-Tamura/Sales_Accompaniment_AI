# server.py
# FastAPI backend for "営業伴走AI"
# - DB: runtime.db is (re)initialized on startup from seed (seed.db / in-code seed)
# - Audio -> Whisper transcription -> (HF diarization if available) -> fallback diarization
# - LLM 3-step pipeline (evidence -> infer -> narratives) + refine via chat
# - Save trigger: persist final minutes, generate DOCX/PDF, then allow moving to Analysis
# - Deals board & deal detail APIs
# - Static hosting for /static and / (index.html)

import os
import io
import re
import json
import shutil
import sqlite3
import uuid
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any, List

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, HTMLResponse, Response
from pydantic import BaseModel, Field

import numpy as np
import soundfile as sf

# --- .env を明示パスで最優先ロード（必ず設定アクセス前に） ---
from dotenv import load_dotenv
_DOTENV_PATH = Path(__file__).resolve().parent / ".env"
load_dotenv(dotenv_path=str(_DOTENV_PATH), override=False, verbose=True)
# --------------------------------------------

# ============ Config ============
BASE_DIR = Path(__file__).resolve().parent
DB_DIR = BASE_DIR / "db"
EXPORT_DIR = BASE_DIR / "exports"
STATIC_DIR = BASE_DIR / "static"
SEED_DB = DB_DIR / "seed.db"        # X (存在すればコピー)

SEED_SQL = DB_DIR / "seed.sql"
SCHEMA_SQL = DB_DIR / "schema.sql"


RUNTIME_DB = DB_DIR / "runtime.db"  # Y (起動時に作り直す)

EXPORT_DIR.mkdir(parents=True, exist_ok=True)
DB_DIR.mkdir(parents=True, exist_ok=True)
STATIC_DIR.mkdir(parents=True, exist_ok=True)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
HF_TOKEN = os.getenv("HF_TOKEN", "")  # optional

MODEL_TEXT = os.getenv("OPENAI_TEXT_MODEL", "gpt-4o")
MODEL_TRANSCRIBE = os.getenv("OPENAI_TRANSCRIBE_MODEL", "whisper-1")
TEMPERATURE = float(os.getenv("OPENAI_TEMPERATURE", "0.2"))

STAFF = ["田中真奈美", "渡辺徹", "小林恭子", "工藤学", "工藤新一"]

STAGES = [
    "リード",
    "商談",
    "提案・概算条件の提示",
    "申込・審査",
    "稟議書認証",
    "契約手続き",
    "融資実行",
    "失注",
]

def clamp_stage(to: Optional[str], default_from: str) -> str:
    if not to:
        return default_from
    to = str(to).strip()
    if to in STAGES:
        return to
    alias = {
        "提案": "提案・概算条件の提示",
        "概算条件提示": "提案・概算条件の提示",
        "提案条件提示": "提案・概算条件の提示",
        "申込": "申込・審査",
        "審査": "申込・審査",
        "稟議": "稟議書認証",
        "契約": "契約手続き",
        "実行": "融資実行",
    }
    to2 = alias.get(to, to)
    if to2 in STAGES:
        return to2
    # どうしても合わなければ「変更なし」
    return default_from


STAGE_PERCENT = {
    "リード": 0,
    "商談:init": 10,
    "商談:repeat": 25,
    "提案・概算条件の提示": 50,
    "申込・審査": 65,
    "稟議書認証": 80,
    "契約手続き": 95,
    "融資実行": 100,
}

# ============ FastAPI ============
app = FastAPI(title="営業伴走AI Backend", version="0.5")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 本番では適切に制限
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============ Static hosting ============
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
INDEX_FILE = STATIC_DIR / "index.html"

@app.get("/", include_in_schema=False)
async def root():
    if INDEX_FILE.exists():
        return FileResponse(str(INDEX_FILE))
    return HTMLResponse("<h1>static/index.html が見つかりません</h1>", status_code=500)

@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    fav = STATIC_DIR / "favicon.ico"
    if fav.exists():
        return FileResponse(str(fav))
    return Response(status_code=204)

@app.get("/health")
def health():
    return {"ok": True, "model": MODEL_TEXT, "transcribe": MODEL_TRANSCRIBE}

@app.get("/debug/env", include_in_schema=False)
def debug_env():
    here = Path(__file__).resolve().parent
    return {
        "cwd": str(here),
        "dotenv_path": str(_DOTENV_PATH),
        "has_OPENAI_API_KEY": bool(os.getenv("OPENAI_API_KEY")),
        "has_HF_TOKEN": bool(os.getenv("HF_TOKEN")),
    }

# ============ DB DDL ============
DDL = """
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
  schedule_json TEXT DEFAULT '[]',
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
"""

# ============ DB Utils ============
def connect_db() -> sqlite3.Connection:
    con = sqlite3.connect(str(RUNTIME_DB))
    con.row_factory = sqlite3.Row
    return con

def exec_script(conn: sqlite3.Connection, script: str):
    conn.executescript(script)
    conn.commit()

def compute_progress(stage: str, meetings_count: int = 1) -> int:
    if stage == "商談":
        return STAGE_PERCENT["商談:repeat" if (meetings_count or 1) >= 2 else "商談:init"]
    return STAGE_PERCENT.get(stage, 0)

def seed_runtime_db_from_seedfile():
    """
    優先順位:
    1) db/seed.db があればそれを runtime.db にコピー
    2) 無ければ db/schema.sql → db/seed.sql を流して runtime.db を生成
    3) どちらも無ければ、内蔵 DDL + in-code seed（従来の3件）で初期化
    """
    # 1) もし db/seed.db があるなら、それを丸ごとコピー
    if SEED_DB.exists():
        shutil.copyfile(SEED_DB, RUNTIME_DB)
        return

    # 2) schema.sql と seed.sql から生成
    conn = connect_db()
    try:
        if SCHEMA_SQL.exists():
            # schema.sql を全文実行（IF NOT EXISTS であれば多重実行も安全）
            exec_script(conn, SCHEMA_SQL.read_text(encoding="utf-8"))
        else:
            # フォールバック：内蔵DDL
            exec_script(conn, DDL)

        if SEED_SQL.exists():
            # seed.sql を全文実行（INSERT群）
            exec_script(conn, SEED_SQL.read_text(encoding="utf-8"))
            return

        # 3) 最終フォールバック：in-code seed（従来の3件）
        deals_seed = [
            ("DEAL-QRS","QRS物流","高橋","商談",120000000,["要注意","大型案件"],"2025-08-01",
             {"格付":"A-","LTV":"62%","DSCR":"1.8x"},["経営者","CFO","経理"],["9/02 条件提示ドラフト共有","9/06 事前相談","9/12 CFOレビュー"], compute_progress("商談",2),0,None),
            ("DEAL-DEF","DEF製作所","佐藤","商談", 38000000,[], "2025-07-28",
             {"格付":"BBB","LTV":"70%"},["営業部長"],[], compute_progress("商談",1),0,None),
            ("DEAL-OPQ","OPQ食品","中村","申込・審査", 54000000,["要注意"],"2025-08-07",
             {"格付":"BBB+"},["購買部長","法務"],[], compute_progress("申込・審査",1),0,None),
        ]
        for row in deals_seed:
            conn.execute(
                """INSERT INTO deals(
                    deal_id,company,owner,stage,amount,labels_json,start_date,
                    credit_json,stakeholders_json,schedule_json,progress_pct,is_lost,last_stage_before_lost
                ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (row[0], row[1], row[2], row[3], row[4],
                 json.dumps(row[5], ensure_ascii=False),
                 row[6],
                 json.dumps(row[7], ensure_ascii=False),
                 json.dumps(row[8], ensure_ascii=False),
                 json.dumps(row[9], ensure_ascii=False),
                 row[10], row[11], row[12])
            )
        conn.commit()
    finally:
        conn.close()
        
def reset_runtime_db():
    if RUNTIME_DB.exists():
        RUNTIME_DB.unlink()
    # ← ここでDDLは流さず、seed関数に一任
    seed_runtime_db_from_seedfile()


@app.on_event("startup")
def on_startup():
    reseed = os.getenv("RESEED", "0")
    if reseed == "1":
        reset_runtime_db()
    else:
        # 初回起動（runtime.db 不在）のときだけ seed を適用
        if not RUNTIME_DB.exists():
            seed_runtime_db_from_seedfile()
        else:
            # 既存DBがあるときはDDLだけ流してスキーマを揃える（IF NOT EXISTS なので安全）
            with connect_db() as conn:
                exec_script(conn, DDL)


# ============ OpenAI Utils ============
def get_openai_client():
    if not OPENAI_API_KEY:
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY not set")
    from openai import OpenAI
    return OpenAI(api_key=OPENAI_API_KEY)

def call_llm_json(prompt: str) -> Dict[str, Any]:
    client = get_openai_client()
    # Responses API (優先)
    try:
        resp = client.responses.create(
            model=MODEL_TEXT,
            temperature=TEMPERATURE,
            response_format={"type": "json_object"},
            input=[
                {"role": "system", "content": "日本語で、必ず有効なJSONのみを出力してください。"},
                {"role": "user", "content": prompt},
            ],
        )
        text = getattr(resp, "output_text", None)
        if text is None:
            # 念のためのフォールバック
            if getattr(resp, "choices", None):
                text = resp.choices[0].message.content
            else:
                text = str(resp)
        return json.loads(text)
    except Exception:
        # ChatCompletions フォールバック
        cc = client.chat.completions.create(
            model=MODEL_TEXT,
            temperature=TEMPERATURE,
            response_format={"type":"json_object"},
            messages=[
                {"role":"system","content":"日本語で、必ず有効なJSONのみを出力してください。"},
                {"role":"user","content":prompt},
            ],
        )
        return json.loads(cc.choices[0].message.content)

# ============ Whisper ============
def _seg_get(seg, key, default=None):
    """seg が dict でもオブジェクトでも .get/.attr 双方に対応"""
    if isinstance(seg, dict):
        return seg.get(key, default)
    return getattr(seg, key, default)

def transcribe_whisper(file_bytes: bytes, filename: str) -> Dict[str, Any]:
    """
    Returns dict: { 'text': str, 'segments': [ { 'start': float, 'end': float, 'text': str } ... ] }
    """
    client = get_openai_client()
    try:
        # Whisper は file-like を要求するので一時保存
        tmp = BASE_DIR / f"_tmp_{uuid.uuid4().hex}_{filename}"
        tmp.write_bytes(file_bytes)
        with open(tmp, "rb") as f:
            tr = client.audio.transcriptions.create(
                model=MODEL_TRANSCRIBE,
                file=f,
                response_format="verbose_json"  # segments 含む
            )
        tmp.unlink(missing_ok=True)

        # tr は pydantic-like object のことがある
        text = getattr(tr, "text", None)
        if text is None and isinstance(tr, dict):
            text = tr.get("text", "")
        segments_raw = getattr(tr, "segments", None)
        if segments_raw is None and isinstance(tr, dict):
            segments_raw = tr.get("segments", [])

        segments: List[Dict[str, Any]] = []
        for s in segments_raw or []:
            start = float(_seg_get(s, "start", 0.0))
            end = float(_seg_get(s, "end", start + 2.0))
            txt = (_seg_get(s, "text", "") or "").strip()
            segments.append({"start": start, "end": end, "text": txt})

        return {"text": text or "", "segments": segments}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"whisper transcription failed: {e}")

# ============ Diarization ============
def diarize_with_pyannote(wav_bytes: bytes) -> Optional[List[Dict[str, Any]]]:
    """
    Try pyannote if HF token exists. Return list of segments:
    [ { 'start': float, 'end': float, 'speaker': 'SPEAKER_00' }, ... ]
    """
    if not HF_TOKEN:
        return None
    try:
        from pyannote.audio import Pipeline
        pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization", use_auth_token=HF_TOKEN)
        # temp wav
        tmpwav = BASE_DIR / f"_tmp_{uuid.uuid4().hex}.wav"
        tmpwav.write_bytes(wav_bytes)
        diarization = pipeline(str(tmpwav))
        tmpwav.unlink(missing_ok=True)
        segs = []
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            segs.append({
                "start": float(turn.start),
                "end": float(turn.end),
                "speaker": speaker
            })
        segs.sort(key=lambda x: x["start"])
        return segs
    except Exception:
        return None

def fallback_diarize_from_transcript_segments(segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    超簡易：無音ギャップと交互で SPEAKER_00/01 を割当
    """
    out = []
    current = "SPEAKER_00"
    last_end = None
    for i, s in enumerate(segments):
        start = float(s.get("start", 0.0))
        end = float(s.get("end", start + 2.0))
        text = s.get("text","").strip()
        if last_end is not None and (start - last_end) > 3.0:
            current = "SPEAKER_01" if current == "SPEAKER_00" else "SPEAKER_00"
        if i % 3 == 0 and i > 0:
            current = "SPEAKER_01" if current == "SPEAKER_00" else "SPEAKER_00"
        out.append({"start": start, "end": end, "speaker": current, "text": text})
        last_end = end
    return out

def attach_speaker_names_with_llm(diarized_turns: List[Dict[str, Any]], owner_bank: str, owner_client: str) -> List[Dict[str, Any]]:
    """
    LLM で SPEAKER_00/01 → 人名推定、だめなら規則で
    """
    try:
        sample_lines = []
        for t in diarized_turns[:12]:
            at = f"{int(t['start']//60):02d}:{int(t['start']%60):02d}"
            sample_lines.append(f"[{at}] {t['speaker']}: {t.get('text','')}")
        prompt = f"""
以下は商談の抜粋です。話者ラベル(SPEAKER_00/01等)を「銀行側:{owner_bank}」「顧客側:{owner_client}」に対応付けてください。
返答は JSON のみ、キーは "map": {{"SPEAKER_00":"...", "SPEAKER_01":"..."}} の形。
<EXCERPT>
{chr(10).join(sample_lines)}
</EXCERPT>
- JSONのみ出力。
""".strip()
        data = call_llm_json(prompt)
        mapping = data.get("map") or {}
        name0 = mapping.get("SPEAKER_00", owner_bank or "銀行側")
        name1 = mapping.get("SPEAKER_01", owner_client or "顧客側")

        named = []
        for t in diarized_turns:
            sp = t["speaker"]
            nm = mapping.get(sp, name0 if sp.endswith("00") else name1)
            named.append({
                "start": t["start"], "end": t["end"],
                "speaker_label": sp, "speaker_name": nm,
                "text": t.get("text","")
            })
        return named
    except Exception:
        named = []
        for t in diarized_turns:
            nm = owner_bank if t["speaker"].endswith("00") else owner_client
            if not nm:
                nm = "銀行側" if t["speaker"].endswith("00") else "顧客側"
            named.append({
                "start": t["start"], "end": t["end"],
                "speaker_label": t["speaker"], "speaker_name": nm,
                "text": t.get("text","")
            })
        return named

# ============ LLM Prompts ============
def _ymd_parts(date_str: str):
    try:
        dt = datetime.fromisoformat(date_str)
        return dt.year, dt.month, dt.day
    except Exception:
        try:
            y, m, d = date_str.split("-")
            return int(y), int(m), int(d)
        except Exception:
            now = datetime.utcnow()
            return now.year, now.month, now.day

def build_prompt_evidence(transcript_text: str, ctx: Dict[str, Any]) -> str:
    return f"""
以下は営業の商談会話録です。日本語で回答し、**厳密なJSONのみ**を出力してください。
目的は、**事実抽出**です。一般語（与信/稟議/担保/申込等）だけではイベントやリスクとしないでください。**前後の文脈**を含む抜粋で根拠を示してください。

# 取引コンテキスト
- 会社名: {ctx.get('company')}
- 担当（銀行）: {ctx.get('owner_bank')}
- 担当（顧客）: {ctx.get('owner_client')}
- 予定取扱額: {ctx.get('amount')}
- 直前ステージ: {ctx.get('prev_stage')}
- 商談回数: {ctx.get('meetings_count')}

# 出力スキーマ
{{
  "turns": [
    {{"speaker":"銀行側|顧客側|CFO|法務|…","text":"一発話","at":"[HH:]MM:SS 可能なら null可"}}
  ],
  "events": [
    {{"type":"提案合意|申込言及|決裁者確定|稟議時期合意|契約合意|融資実行合意|先延ばし言及|予算不足言及|競合優位言及|条件不一致|与信/担保障壁言及",
      "evidence_span":"turnを2〜6連結した文脈抜粋（複数話者可）",
      "turn_indices":[開始index, 終了index]}}
  ],
  "quotes_for_stage": [
    {{"speaker":"…","quote":"ステージ変更の根拠として十分な1〜3文","at":"[HH:]MM:SS 可能なら"}}
  ],
  "risk_candidates": [
    {{"category":"決裁不在|価格|予算|与信|担保|タイムライン後ろ倒し|関係悪化|競合優位|その他",
      "severity_proposal":"High|Medium|Low",
      "evidence_span":"前後の文脈を含む抜粋（2〜6行）",
      "at":"[HH:]MM:SS 可能なら",
      "notes":"必要なら補足"}}
  ]
}}

<TRANSCRIPT>
{transcript_text}
</TRANSCRIPT>
- JSON以外は出力しない。
""".strip()

def build_prompt_infer(evidence_json: Dict[str, Any], ctx: Dict[str, Any]) -> str:
    return f"""
以下は事実抽出のJSONです。これと取引コンテキストを踏まえ、変更案とリスクを推論し、同一スキーマのJSONのみを出力してください。

# 取引コンテキスト
- 直前ステージ: {ctx.get('prev_stage')}
- 商談回数: {ctx.get('meetings_count')}

# 入力: evidence_json
{json.dumps(evidence_json, ensure_ascii=False)}

# 出力スキーマ
{{
  "suggested_changes": {{
    "stage": {{"from": "{ctx.get('prev_stage')}", "to": "<次ステージ or 同じ>", "confidence": 0.0, "rationale": "根拠の要約（1文）"}},
    "labels_add": [],
    "labels_remove": [],
    "owner_change": null,
    "amount_change": null
  }},
  "stage_change_evidence": [{{"speaker":"…","quote":"…","at":"…"}}],
  "risks": [{{"category":"…","severity":"High|Medium|Low","evidence_span":"文脈抜粋","context_summary":"全体的な文脈の要約","at":"…","action":"1文"}}]
}}
- ルール: 単語出現のみでリスク化しない。重大度 High または Medium 複数なら labels_add に "要注意" を含める。stage.to は {STAGES} のいずれか**1つ**に限定し、変更が無い場合は from と同じ値にする（「同じ」という語は使わない）。
- JSON以外は出力しない。
""".strip()

def build_prompt_narratives(evidence_json: Dict[str, Any], infer_json: Dict[str, Any], ctx: Dict[str, Any], minutes_hint: str = "") -> str:
    y, m, d = _ymd_parts(ctx.get("meeting_date",""))
    # 指定テンプレート（ご要望どおり）
    template = (
        "商談議事録\n\n"
        "1. 商談概要\n"
        f"日時：{y}年 {int(m)}月 {int(d)}日\n"
        f"取引先：{ctx.get('company')}\n"
        "出席者\n"
        f"  銀行側：{ctx.get('owner_bank')}\n"
        f"  顧客側：{ctx.get('owner_client')}\n"
        "\n"
        "2. 議論の要約\n"
        "・（要点1）\n"
        "・（要点2）\n"
        "\n"
        "3. 決定事項\n"
        "・（決定1）\n"
        "\n"
        "4. ToDO\n"
        "・（誰が／いつまでに／何を）\n"
        "\n"
        "5. 懸念点（商談におけるリスク）\n"
        "・（該当なしの場合は「特記なし」）\n"
    )
    hint = f"- 参考: ユーザー編集ドラフトを尊重して統合してください。\n{minutes_hint}" if minutes_hint else ""
    return f"""
以下の事実抽出(evidence)と推論(infer)を元に、文面を作成してください。日本語、JSONのみ。
- 出力は次のテンプレートに沿った Markdown を minutes_draft に収めてください。

<MINUTES_TEMPLATE>
{template}
</MINUTES_TEMPLATE>

- 会社名: {ctx.get('company')} / 担当(銀): {ctx.get('owner_bank')} / 担当(顧): {ctx.get('owner_client')} / 金額: {ctx.get('amount')} / 日時: {ctx.get('meeting_date')}

# evidence_json
{json.dumps(evidence_json, ensure_ascii=False)}

# infer_json
{json.dumps(infer_json, ensure_ascii=False)}

# 出力スキーマ
{{
  "minutes_draft": "Markdown 文字列",
  "manager_report_draft": "上司向けに200〜350字で、現状/懸念/次アクションを簡潔に（箇条書き禁止）",
  "next_actions": ["アクション1", "アクション2"]
}}
{hint}
- JSON以外は出力しない。
""".strip()

def build_prompt_refine(transcript_text: str, current_json: Dict[str, Any], user_instruction: str) -> str:
    return f"""
あなたは編集アシスタントです。**同じスキーマ**のJSONのみを出力してください。
- 目的: ユーザーの指示に従い、変更プレビュー/リスク/上司レポ/議事録を更新する。
- 注意: キー名は維持。必要に応じてエビデンスも更新。

# ユーザー指示
{user_instruction}

# 現在のJSON
{json.dumps(current_json, ensure_ascii=False)}

# 参考（全文）
<TRANSCRIPT>
{transcript_text}
</TRANSCRIPT>

# 出力スキーマ
{{
  "minutes_draft": "Markdown 文字列（省略可）",
  "manager_report_draft": "上司向けテキスト（省略可）",
  "next_actions": ["..."],
  "risks": [...],
  "stage_change_evidence": [...],
  "suggested_changes": {{}},
  "coach_feedback": "会話とフェーズに即したコーチング（省略可）"
}}

# 出力
- JSON以外は出力しない。
""".strip()


def build_prompt_coach(evidence_json: Dict[str, Any], infer_json: Dict[str, Any], ctx: Dict[str, Any], transcript_text: str) -> str:
    """
    会話データとフェーズ（prev→suggested）を踏まえた営業コーチング文面（日本語）を生成。
    フロントの簡易テンプレの趣旨をプロンプト内で再利用する。
    """
    prev_stage = ctx.get("prev_stage", "商談")
    sug = (infer_json or {}).get("suggested_changes", {}).get("stage", {}) or {}
    to_stage = sug.get("to") or prev_stage
    stage_key = f"{prev_stage}→{to_stage if to_stage != prev_stage else '同じ'}"

    # 簡易テンプレを要約してLLMに渡す（フロントのロジックの主旨を再利用）
    phase_guidelines = {
        "商談→同じ": "意思決定者/評価軸/タイムラインの具体化。次回提案ドラフト合意。例: 次回◯日までに概算条件ドラフト共有→レビュー合意を取る。",
        "商談→提案・概算条件の提示": "価格だけでなく効果/リスク低減/意思決定適合性の比較軸。申込・審査に向け必要資料を先出し。例: 提案ドラフト提示→審査必要資料の事前準備合意。",
        "提案・概算条件の提示→申込・審査": "審査移行の阻害要因（体制/書類/稟議順序）を先回りで解消。例: 直近試算表や◯◯実績の先行提供依頼。",
        "申込・審査→稟議書認証": "リスクへの対案（担保/保証/コベナンツ）を明示し社内稟議の論点を前倒しで潰す。",
        "稟議書認証→契約手続き": "誰が/いつ/何をサインするか段取り確認でリードタイム短縮。",
        "契約手続き→融資実行": "実行条件のToDo（登記/入金口座/日程）を明確化し逆算スケジュールを提示。"
    }

    return f"""
以下は営業商談データです。日本語で、**JSONのみ**を出力してください。

# フェーズ
- 直前ステージ: {prev_stage}
- 提案ステージ: {to_stage}
- キー: {stage_key}

# テンプレ主旨（再利用）
- {json.dumps(phase_guidelines, ensure_ascii=False)}

# 根拠データ
- evidence_json: {json.dumps(evidence_json, ensure_ascii=False)}
- infer_json: {json.dumps(infer_json, ensure_ascii=False)}

# 会話（整形済み）
<TRANSCRIPT>
{transcript_text}
</TRANSCRIPT>

# 出力スキーマ
{{
  "coach_feedback": "250〜450字で、今回の会話の内容に即した具体的な改善アドバイス。フェーズ遷移の観点（{stage_key}）を踏まえ、次回の合意・必要資料・言い回し例を最低1つ含める。箇条書きは可、丁寧語。"
}}

- JSON以外は出力しない。
""".strip()


# ============ Helpers ============
def make_deal_id(company: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9]+", "", company)[:6].upper()
    return f"DEAL-{slug or uuid.uuid4().hex[:6]}"

def make_meeting_id(company: str, meeting_date: str) -> str:
    return f"MEET-{uuid.uuid4().hex[:8]}"

def join_turns_text(turns: List[Dict[str, Any]]) -> str:
    lines = []
    for t in turns:
        at = f"{int(t['start']//60):02d}:{int(t['start']%60):02d}"
        lines.append(f"[{at}] {t.get('speaker_name', t.get('speaker_label',''))}: {t.get('text','')}")
    return "\n".join(lines)

def minutes_new_version(conn: sqlite3.Connection, meeting_id: str, minutes_md: str, manager_report: str, next_actions: List[str]) -> int:
    cur = conn.execute("SELECT COALESCE(MAX(version),0) FROM minutes WHERE meeting_id=?", (meeting_id,))
    ver = (cur.fetchone() or [0])[0] + 1
    conn.execute("UPDATE minutes SET is_current=0 WHERE meeting_id=?", (meeting_id,))
    conn.execute(
        "INSERT INTO minutes(id, meeting_id, version, is_current, minutes_md, manager_report, next_actions_json, docx_path, pdf_path, created_at) VALUES(?,?,?,?,?,?,?,?,?,?)",
        (uuid.uuid4().hex, meeting_id, ver, 1, minutes_md, manager_report, json.dumps(next_actions, ensure_ascii=False), "", "", datetime.utcnow().isoformat())
    )
    conn.commit()
    return ver

def risks_replace(conn: sqlite3.Connection, meeting_id: str, risks: List[Dict[str, Any]]):
    conn.execute("DELETE FROM risks WHERE meeting_id=?", (meeting_id,))
    for r in risks or []:
        conn.execute(
            "INSERT INTO risks(id,meeting_id,category,severity,evidence_span,context_summary,at,action) VALUES(?,?,?,?,?,?,?,?)",
            (uuid.uuid4().hex, meeting_id, r.get("category",""), r.get("severity",""), r.get("evidence_span",""), r.get("context_summary",""), r.get("at",""), r.get("action",""))
        )
    conn.commit()

def stage_evidence_replace(conn: sqlite3.Connection, meeting_id: str, evidences: List[Dict[str, Any]]):
    conn.execute("DELETE FROM stage_evidence WHERE meeting_id=?", (meeting_id,))
    for e in evidences or []:
        conn.execute(
            "INSERT INTO stage_evidence(id,meeting_id,speaker_name,quote,at) VALUES(?,?,?,?,?)",
            (uuid.uuid4().hex, meeting_id, e.get("speaker",""), e.get("quote",""), e.get("at",""))
        )
    conn.commit()

def render_docx(path: Path, minutes_md: str):
    from docx import Document
    doc = Document()
    for line in minutes_md.splitlines():
        doc.add_paragraph(line)
    doc.save(str(path))

def render_pdf(path: Path, minutes_md: str):
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen import canvas
    from reportlab.lib.units import mm
    c = canvas.Canvas(str(path), pagesize=A4)
    width, height = A4
    x = 20*mm
    y = height - 20*mm
    for line in minutes_md.splitlines():
        if y < 20*mm:
            c.showPage()
            y = height - 20*mm
        c.drawString(x, y, line)
        y -= 6*mm
    c.save()

# ============ Pydantic Models ============
class RunTextPayload(BaseModel):
    company: str
    meeting_date: str
    prev_stage: str
    amount: Optional[int] = None
    owner_bank: str
    owner_client: str
    meetings_count: int = 1
    transcript_text: str
    minutes_hint: Optional[str] = ""

class RefinePayload(BaseModel):
    meeting_id: str
    user_instruction: Optional[str] = None
    instruction: Optional[str] = None

    @property
    def text(self) -> str:
        return (self.user_instruction or self.instruction or "").strip()


class SaveFinalPayload(BaseModel):
    meeting_id: str
    minutes_md: str
    selected_stage: str
    labels: List[str] = Field(default_factory=list)

# ============ API: Dashboard (Kanban) ============
@app.get("/api/dashboard/list")
def api_dashboard_list(owner: str = "", labels: str = ""):
    """
    返却形式：{ "columns": { "<stage>": [ {deal_id, company, owner, labels[], progress_pct, meeting_date, ...}, ... ] } }
    ※ meeting_date は meetings テーブルから会社ごとの最新日付を集計して付与
    """
    with connect_db() as con:
        q = "SELECT * FROM deals"
        params: List[Any] = []
        where: List[str] = []
        if owner:
            where.append("owner = ?")
            params.append(owner)
        if labels:
            # labels は CSV。「AND 全含有」より「OR 含有」寄りの検索だが、現状は LIKE AND のシンプル運用
            for lb in [s.strip() for s in labels.split(",") if s.strip()]:
                where.append("labels_json LIKE ?")
                params.append(f"%{lb}%")
        if where:
            q += " WHERE " + " AND ".join(where)
        q += " ORDER BY created_at DESC"

        deals = con.execute(q, params).fetchall()

        # 会社ごとの最新 meeting_date をまとめて取得
        recent_rows = con.execute(
            "SELECT company, MAX(meeting_date) AS recent_dt FROM meetings GROUP BY company"
        ).fetchall()
        recent_map = {r["company"]: (r["recent_dt"] or "") for r in recent_rows}

        grouped: Dict[str, List[Dict[str, Any]]] = {st: [] for st in STAGES}
        for r in deals:
            labels_list = json.loads(r["labels_json"]) if r["labels_json"] else []
            credit = json.loads(r["credit_json"]) if r["credit_json"] else {}
            stakeholders = json.loads(r["stakeholders_json"]) if r["stakeholders_json"] else []
            company = r["company"]
            # 最新 meeting_date（無ければ start_date）
            md = recent_map.get(company) or (r["start_date"] or "")

            grouped[r["stage"]].append({
                "deal_id": r["deal_id"],
                "company": company,
                "owner": r["owner"] or "",
                "amount": r["amount"],
                "start_date": r["start_date"] or "",
                "meeting_date": md,  # ★ 追加
                "labels": labels_list,
                "credit": " / ".join(f"{k}{'' if v=='' else ' '}{v}" for k,v in credit.items()) if credit else "—",
                "stakeholders": stakeholders,
                "progress_pct": r["progress_pct"] or 0,
            })
        return {"columns": grouped}


# KPI（ダミー）
@app.get("/api/metrics/performance")
def api_metrics_performance():
    with connect_db() as con:
        rows = con.execute("SELECT stage, amount FROM deals").fetchall()
        total = sum(r["amount"] or 0 for r in rows if r["stage"] != "失注")
        return {
            "total_amount_this_month": total,
            "target_achievement_pct": 104,
            "executed_count": 62,
            "delta_amount_pct": +8.2,
            "delta_executed": -3,
        }

# ============ API: Deal detail ============
@app.get("/api/deals/{deal_id}")
def api_deal_detail(deal_id: str):
    with connect_db() as con:
        r = con.execute("SELECT * FROM deals WHERE deal_id=?", (deal_id,)).fetchone()
        if not r:
            raise HTTPException(404, "deal not found")

        # 過去 minutes（会社名でひもづけ）
        company = r["company"]
        mins = con.execute(
            """SELECT mi.meeting_id, mi.created_at, substr(mi.minutes_md,1,120) AS excerpt
               FROM minutes mi
               JOIN meetings m ON m.meeting_id = mi.meeting_id
               WHERE m.company=?
               AND mi.is_current=1
               ORDER BY mi.created_at DESC
               LIMIT 5""",
            (company,)
        ).fetchall()
        past = [{"meeting_id":x["meeting_id"], "saved_at":x["created_at"], "excerpt":x["excerpt"]} for x in mins]

        # 予定（ダミー：deals.schedule_json 使用）
        schedules = json.loads(r["schedule_json"]) if r["schedule_json"] else []
        schedules = [{"date":"—","text":s} if isinstance(s,str) else s for s in schedules]

        # 直近の商談日
        recent_row = con.execute(
            "SELECT MAX(meeting_date) AS recent_dt FROM meetings WHERE company=?", (company,)
        ).fetchone()
        recent_meeting_date = (recent_row["recent_dt"] if recent_row and recent_row["recent_dt"] else r["start_date"])

        # 上司レポート＆リスク（最新 meeting を採択）
        last_mid_row = con.execute(
            """SELECT mi.meeting_id
               FROM minutes mi
               JOIN meetings m ON m.meeting_id=mi.meeting_id
               WHERE m.company=?
               ORDER BY mi.created_at DESC
               LIMIT 1""",
            (company,)
        ).fetchone()
        manager_report = ""
        risks = []
        if last_mid_row:
            mid = last_mid_row["meeting_id"]
            mgr = con.execute(
                "SELECT manager_report FROM minutes WHERE meeting_id=? AND is_current=1", (mid,)
            ).fetchone()
            if mgr:
                manager_report = mgr["manager_report"] or ""
            rs = con.execute(
                "SELECT category,severity,evidence_span,context_summary,at,action FROM risks WHERE meeting_id=?",
                (mid,)
            ).fetchall()
            risks = [{"category":x[0], "severity":x[1], "evidence_span":x[2], "context_summary":x[3], "at":x[4], "action":x[5]} for x in rs]

        return {
            "deal": {
                "deal_id": r["deal_id"],
                "company": r["company"],
                "owner": r["owner"] or "",
                "amount": r["amount"],
                "start_date": r["start_date"] or "",
                "stage": r["stage"],
                "progress_pct": r["progress_pct"] or 0,
                "labels": json.loads(r["labels_json"]) if r["labels_json"] else [],
                "credit": json.loads(r["credit_json"]) if r["credit_json"] else {},
                "stakeholders": json.loads(r["stakeholders_json"]) if r["stakeholders_json"] else [],
            },
            "past_minutes": past,
            "schedules": schedules,
            "recent_meeting_date": recent_meeting_date or "",
            "manager_report": manager_report,
            "risks": risks
        }

# ============ API: Minutes – run from TEXT ============
@app.post("/api/minutes/run_text")
def api_run_text(payload: RunTextPayload):
    ctx = payload.model_dump()
    amount_val = int(ctx["amount"]) if (ctx.get("amount") is not None) else None

    # テキストから簡易分割（[mm:ss] Speaker: text を優先、なければ行順）
    turns = []
    for i, line in enumerate(ctx["transcript_text"].splitlines()):
        m = re.match(r"\[(\d{1,2}):(\d{2})\]\s*(.+?)\s*[:：]\s*(.+)", line.strip())
        if m:
            mm_, ss_, spk, txt = m.groups()
            start = int(mm_) * 60 + int(ss_)
            turns.append({"start": float(start), "end": float(start + max(2, len(txt)//10)), "speaker": "SPEAKER_00" if i%2==0 else "SPEAKER_01", "text": txt})
        elif line.strip():
            turns.append({"start": float(i*5), "end": float(i*5+3), "speaker": "SPEAKER_00" if i%2==0 else "SPEAKER_01", "text": line.strip()})

    named_turns = attach_speaker_names_with_llm(turns, ctx["owner_bank"], ctx["owner_client"])

    meeting_id = make_meeting_id(ctx["company"], ctx["meeting_date"])
    with connect_db() as conn:
        conn.execute(
            "INSERT INTO meetings(meeting_id,company,meeting_date,owner_bank,owner_client,prev_stage,amount,meetings_count,transcript_raw,diarized_json,selected_stage,labels_json) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
            (meeting_id, ctx["company"], ctx["meeting_date"], ctx["owner_bank"], ctx["owner_client"],
             ctx["prev_stage"], amount_val, ctx["meetings_count"],
             ctx["transcript_text"], json.dumps(named_turns, ensure_ascii=False),
             ctx["prev_stage"], json.dumps([], ensure_ascii=False))
        )
        for idx, t in enumerate(named_turns):
            conn.execute(
                "INSERT INTO transcript_turns(id,meeting_id,turn_index,speaker_label,speaker_name,start_sec,end_sec,text) VALUES(?,?,?,?,?,?,?,?)",
                (uuid.uuid4().hex, meeting_id, idx, t["speaker_label"], t["speaker_name"], t["start"], t["end"], t["text"])
            )
        conn.commit()

    # 3-step LLM
    transcript_for_llm = join_turns_text(named_turns)
    ev = call_llm_json(build_prompt_evidence(transcript_for_llm, ctx))
    inf = call_llm_json(build_prompt_infer(ev, ctx))
    # ★ 追加（LLM出力を強制的に正規化）
    st = ((inf.get("suggested_changes") or {}).get("stage") or {})
    st["to"] = clamp_stage(st.get("to"), ctx["prev_stage"])
    inf.setdefault("suggested_changes", {})["stage"] = st
    nar = call_llm_json(build_prompt_narratives(ev, inf, ctx, minutes_hint=ctx.get("minutes_hint","")))

    # ★ 追加：コーチング（会話＋フェーズに基づく）
    coach_out = call_llm_json(build_prompt_coach(ev, inf, ctx, transcript_for_llm))
    coach_feedback = coach_out.get("coach_feedback", "")


    with connect_db() as conn:
        minutes_new_version(conn, meeting_id, nar.get("minutes_draft",""), nar.get("manager_report_draft",""), nar.get("next_actions",[]))
        risks_replace(conn, meeting_id, inf.get("risks",[]))
        stage_evidence_replace(conn, meeting_id, inf.get("stage_change_evidence",[]))

    # UI 即時表示用の簡易 turns（at/speaker）
    turns_for_ui = []
    for t in named_turns:
        at = f"{int(t['start']//60):02d}:{int(t['start']%60):02d}"
        turns_for_ui.append({"at": at, "speaker": t["speaker_name"], "text": t["text"]})

    return {
        "meeting_id": meeting_id,
        "turns": turns_for_ui,
        "evidence": ev,
        "infer": inf,
        "narratives": nar,
        "coach_feedback": coach_feedback  # ★ 追加
    }

# ============ API: Minutes – run from AUDIO ============
@app.post("/api/minutes/run_audio")
async def api_run_audio(
    company: str = Form(...),
    meeting_date: str = Form(...),
    prev_stage: str = Form(...),
    owner_bank: str = Form(...),
    owner_client: str = Form(...),
    meetings_count: int = Form(1),
    amount: Optional[int] = Form(None),
    audio: UploadFile = File(...)
):
    file_bytes = await audio.read()
    amount_val = int(amount) if (amount is not None) else None

    # Whisper
    tr = transcribe_whisper(file_bytes, audio.filename)
    segments = tr.get("segments", [])
    if not segments:
        raise HTTPException(status_code=500, detail="transcription returned empty segments")

    # diarize：pyannote 優先（wavに正規化）
    diar_segs = None
    try:
        data, sr = sf.read(io.BytesIO(file_bytes))
        wav_buf = io.BytesIO()
        sf.write(wav_buf, data, sr, format="WAV")
        diar_segs = diarize_with_pyannote(wav_buf.getvalue())
    except Exception:
        diar_segs = None

    if diar_segs is None:
        diarized = fallback_diarize_from_transcript_segments(segments)
    else:
        # 時間でざっくり紐づけてテキストを集約
        diarized = []
        j = 0
        for ds in diar_segs:
            chunk_texts = []
            while j < len(segments) and float(segments[j]["start"]) < ds["end"]:
                if float(segments[j]["end"]) > ds["start"]:
                    chunk_texts.append(segments[j]["text"])
                j += 1
            diarized.append({
                "start": ds["start"], "end": ds["end"], "speaker": ds["speaker"],
                "text": " ".join(chunk_texts).strip()
            })

    named_turns = attach_speaker_names_with_llm(diarized, owner_bank, owner_client)

    ctx = {
        "company": company, "meeting_date": meeting_date, "prev_stage": prev_stage,
        "amount": amount_val, "owner_bank": owner_bank, "owner_client": owner_client,
        "meetings_count": meetings_count
    }

    meeting_id = make_meeting_id(company, meeting_date)
    with connect_db() as conn:
        conn.execute(
            "INSERT INTO meetings(meeting_id,company,meeting_date,owner_bank,owner_client,prev_stage,amount,meetings_count,transcript_raw,diarized_json,selected_stage,labels_json) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
            (meeting_id, company, meeting_date, owner_bank, owner_client,
             prev_stage, amount_val, meetings_count,
             tr.get("text",""), json.dumps(named_turns, ensure_ascii=False),
             prev_stage, json.dumps([], ensure_ascii=False))
        )
        for idx, t in enumerate(named_turns):
            conn.execute(
                "INSERT INTO transcript_turns(id,meeting_id,turn_index,speaker_label,speaker_name,start_sec,end_sec,text) VALUES(?,?,?,?,?,?,?,?)",
                (uuid.uuid4().hex, meeting_id, idx, t["speaker_label"], t["speaker_name"], t["start"], t["end"], t["text"])
            )
        conn.commit()

    # 3-step LLM
    transcript_for_llm = join_turns_text(named_turns)
    ev = call_llm_json(build_prompt_evidence(transcript_for_llm, ctx))
    inf = call_llm_json(build_prompt_infer(ev, ctx))
    # ★ 追加
    st = ((inf.get("suggested_changes") or {}).get("stage") or {})
    st["to"] = clamp_stage(st.get("to"), prev_stage)
    inf.setdefault("suggested_changes", {})["stage"] = st

    nar = call_llm_json(build_prompt_narratives(ev, inf, ctx))

    # ★ 追加：コーチング
    coach_out = call_llm_json(build_prompt_coach(ev, inf, ctx, transcript_for_llm))
    coach_feedback = coach_out.get("coach_feedback", "")


    with connect_db() as conn:
        minutes_new_version(conn, meeting_id, nar.get("minutes_draft",""), nar.get("manager_report_draft",""), nar.get("next_actions",[]))
        risks_replace(conn, meeting_id, inf.get("risks",[]))
        stage_evidence_replace(conn, meeting_id, inf.get("stage_change_evidence",[]))

    turns_for_ui = []
    for t in named_turns:
        at = f"{int(t['start']//60):02d}:{int(t['start']%60):02d}"
        turns_for_ui.append({"at": at, "speaker": t["speaker_name"], "text": t["text"]})

    return {
        "meeting_id": meeting_id,
        "turns": turns_for_ui,
        "evidence": ev,
        "infer": inf,
        "narratives": nar,
        "coach_feedback": coach_feedback  # ★ 追加
    }

# ============ API: refine chat ============
@app.post("/api/minutes/refine_chat")
def api_refine(payload: RefinePayload):
    instruction = payload.text
    if not instruction:
        raise HTTPException(status_code=400, detail="user_instruction is required")
    # 以降は既存処理でOK（build_prompt_refine 修正済みであれば 500 は解消）

    with connect_db() as conn:
        row = conn.execute("SELECT diarized_json FROM meetings WHERE meeting_id=?", (payload.meeting_id,)).fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="meeting not found")
        diarized = json.loads(row[0]) if isinstance(row[0], str) else (row[0] or [])
        transcript_text = join_turns_text(diarized)

        # current json を構築
        cur_stage_evi = conn.execute("SELECT speaker_name,quote,at FROM stage_evidence WHERE meeting_id=?", (payload.meeting_id,)).fetchall()
        cur_risks = conn.execute("SELECT category,severity,evidence_span,context_summary,at,action FROM risks WHERE meeting_id=?", (payload.meeting_id,)).fetchall()
        cur_min = conn.execute("SELECT minutes_md, manager_report, next_actions_json FROM minutes WHERE meeting_id=? AND is_current=1", (payload.meeting_id,)).fetchone()

        current_json = {
            "suggested_changes": {},
            "stage_change_evidence": [{"speaker":r[0],"quote":r[1],"at":r[2]} for r in cur_stage_evi],
            "risks": [{"category":r[0],"severity":r[1],"evidence_span":r[2],"context_summary":r[3],"at":r[4],"action":r[5]} for r in cur_risks],
            "minutes_draft": (cur_min[0] if cur_min else ""),
            "manager_report_draft": (cur_min[1] if cur_min else ""),
            "next_actions": (json.loads(cur_min[2]) if cur_min and cur_min[2] else []),
            "coach_feedback": ""  # ★ 追加（現状は未保存。都度再生成/更新想定）
        }


        # （既存）out を作る直前で、DBから prev/selected を取っておく
        row2 = conn.execute(
            "SELECT prev_stage, selected_stage FROM meetings WHERE meeting_id=?",
            (payload.meeting_id,)
        ).fetchone()
        state_prev = (row2["selected_stage"] or row2["prev_stage"] or "商談") if row2 else "商談"



        out = call_llm_json(build_prompt_refine(transcript_text, current_json, instruction))

        # ★ 追加（正規化）
        st = ((out.get("suggested_changes") or {}).get("stage") or {})
        st["to"] = clamp_stage(st.get("to"), state_prev)
        out.setdefault("suggested_changes", {})["stage"] = st

        # 保存
        minutes_md = out.get("minutes_draft", current_json["minutes_draft"])
        manager_report = out.get("manager_report_draft", current_json["manager_report_draft"])
        next_actions = out.get("next_actions", current_json["next_actions"])
        ver = minutes_new_version(conn, payload.meeting_id, minutes_md, manager_report, next_actions)

        # 置換
        if "risks" in out:
            risks_replace(conn, payload.meeting_id, out.get("risks",[]))
        if "stage_change_evidence" in out:
            stage_evidence_replace(conn, payload.meeting_id, out.get("stage_change_evidence",[]))

        conn.execute("INSERT INTO chat_edits(id,meeting_id,user_text,applied_version) VALUES(?,?,?,?)",
                     (uuid.uuid4().hex, payload.meeting_id, payload.user_instruction, ver))
        conn.commit()

        return {"ok": True, "version": ver, "data": out}

# ============ API: save final (trigger) ============
@app.post("/api/minutes/save_final")
def api_save_final(payload: SaveFinalPayload):
    with connect_db() as conn:
        m = conn.execute(
            "SELECT company, meeting_date, owner_bank, owner_client, prev_stage, amount, meetings_count FROM meetings WHERE meeting_id=?",
            (payload.meeting_id,)
        ).fetchone()
        if not m:
            raise HTTPException(status_code=404, detail="meeting not found")

        # 現行 minutes の上書き保存（バージョン更新）
        cur_min = conn.execute("SELECT manager_report, next_actions_json FROM minutes WHERE meeting_id=? AND is_current=1", (payload.meeting_id,)).fetchone()
        manager_report = (cur_min[0] if cur_min else "")
        next_actions = json.loads(cur_min[1]) if (cur_min and cur_min[1]) else []
        minutes_new_version(conn, payload.meeting_id, payload.minutes_md, manager_report, next_actions)

        # meeting の選択ステージ & ラベルを更新
        conn.execute("UPDATE meetings SET selected_stage=?, labels_json=? WHERE meeting_id=?",
                     (payload.selected_stage, json.dumps(payload.labels, ensure_ascii=False), payload.meeting_id))

        # deals に upsert
        company = m["company"]
        owner_bank = m["owner_bank"]
        amount = m["amount"]
        meetings_count = m["meetings_count"] or 1
        deal_id = make_deal_id(company)
        progress = compute_progress(payload.selected_stage, meetings_count)

        cur = conn.execute("SELECT deal_id FROM deals WHERE deal_id=?", (deal_id,)).fetchone()
        if cur:
            conn.execute(
                "UPDATE deals SET owner=?, stage=?, amount=?, labels_json=?, progress_pct=?, is_lost=?, last_stage_before_lost=?, updated_at=datetime('now') WHERE deal_id=?",
                (owner_bank, payload.selected_stage, amount, json.dumps(payload.labels, ensure_ascii=False),
                 progress, 1 if payload.selected_stage=="失注" else 0,
                 m["meeting_date"] if payload.selected_stage=="失注" else None,
                 deal_id)
            )
        else:
            conn.execute(
                """INSERT INTO deals(deal_id, company, owner, stage, amount, labels_json, start_date, credit_json, stakeholders_json, schedule_json, progress_pct, is_lost, last_stage_before_lost)
                   VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (deal_id, company, owner_bank, payload.selected_stage, amount,
                 json.dumps(payload.labels, ensure_ascii=False),
                 m["meeting_date"], json.dumps({}, ensure_ascii=False),
                 json.dumps([m["owner_client"]], ensure_ascii=False),
                 json.dumps([], ensure_ascii=False),
                 progress, 0, None)
            )

        # エクスポート（docx/pdf）
        docx_path = EXPORT_DIR / f"{payload.meeting_id}.docx"
        pdf_path = EXPORT_DIR / f"{payload.meeting_id}.pdf"
        try:
            render_docx(docx_path, payload.minutes_md)
            render_pdf(pdf_path, payload.minutes_md)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"export failed: {e}")

        conn.execute("UPDATE minutes SET docx_path=?, pdf_path=? WHERE meeting_id=? AND is_current=1",
                     (str(docx_path), str(pdf_path), payload.meeting_id))
        conn.commit()

        return {
            "ok": True,
            "docx_url": f"/api/minutes/docx?meeting_id={payload.meeting_id}",
            "pdf_url": f"/api/minutes/pdf?meeting_id={payload.meeting_id}",
            "allow_next": True,
            "deal_id": deal_id
        }

# ダウンロード（安全ヘッダを追加）
@app.get("/api/minutes/docx")
def api_docx(meeting_id: str = Query(...)):
    with connect_db() as conn:
        row = conn.execute("SELECT docx_path FROM minutes WHERE meeting_id=? AND is_current=1", (meeting_id,)).fetchone()
        if not row or not row[0] or not Path(row[0]).exists():
            raise HTTPException(status_code=404, detail="docx not found")
        return FileResponse(
            row[0],
            filename=f"{meeting_id}.docx",
            media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            headers={"X-Content-Type-Options": "nosniff"}
        )

@app.get("/api/minutes/pdf")
def api_pdf(meeting_id: str = Query(...)):
    with connect_db() as conn:
        row = conn.execute("SELECT pdf_path FROM minutes WHERE meeting_id=? AND is_current=1", (meeting_id,)).fetchone()
        if not row or not row[0] or not Path(row[0]).exists():
            raise HTTPException(status_code=404, detail="pdf not found")
        return FileResponse(
            row[0],
            filename=f"{meeting_id}.pdf",
            media_type="application/pdf",
            headers={"X-Content-Type-Options": "nosniff"}
        )

# 会話ターン取得（分析・コーチング用）
@app.get("/api/meetings/turns")
def api_meeting_turns(meeting_id: str):
    with connect_db() as conn:
        rows = conn.execute(
            "SELECT turn_index,speaker_label,speaker_name,start_sec,end_sec,text FROM transcript_turns WHERE meeting_id=? ORDER BY turn_index ASC",
            (meeting_id,)
        ).fetchall()
        turns = []
        for r in rows:
            at = f"{int((r['start_sec'] or 0)//60):02d}:{int((r['start_sec'] or 0)%60):02d}"
            turns.append({
                "turn_index": r["turn_index"],
                "speaker_label": r["speaker_label"],
                "speaker_name": r["speaker_name"],
                "speaker": r["speaker_name"],   # UI互換
                "start": r["start_sec"],
                "end": r["end_sec"],
                "at": at,
                "text": r["text"]
            })
        return {"turns": turns}

# リスク・上司レポ取得（分析タブ）
@app.get("/api/analysis/risks")
def api_analysis_risks(meeting_id: str):
    with connect_db() as conn:
        risks = conn.execute("SELECT category,severity,evidence_span,context_summary,at,action FROM risks WHERE meeting_id=?", (meeting_id,)).fetchall()
        stage_evi = conn.execute("SELECT speaker_name,quote,at FROM stage_evidence WHERE meeting_id=?", (meeting_id,)).fetchall()
        mgr = conn.execute("SELECT manager_report FROM minutes WHERE meeting_id=? AND is_current=1", (meeting_id,)).fetchone()
        return {
            "risks": [{"category":r[0],"severity":r[1],"evidence_span":r[2],"context_summary":r[3],"at":r[4],"action":r[5]} for r in risks],
            "stage_evidence": [{"speaker":e[0],"quote":e[1],"at":e[2]} for e in stage_evi],
            "manager_report": (mgr[0] if mgr else "")
        }


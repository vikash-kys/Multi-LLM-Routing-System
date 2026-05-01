import sqlite3
import json
import logging
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import Optional

logger = logging.getLogger(__name__)

DB_PATH = Path("logs/requests.db")


@dataclass
class RequestLog:
    id: Optional[int]
    timestamp: str
    prompt_hash: str
    prompt_preview: str          # First 200 chars
    complexity_tier: int
    routed_model: str
    provider: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    latency_ms: int
    quality_score: Optional[float]
    was_escalated: bool
    escalated_model: Optional[str]
    escalation_cost_delta: Optional[float]
    error: Optional[str]


def init_db():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS requests (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            prompt_hash TEXT NOT NULL,
            prompt_preview TEXT,
            complexity_tier INTEGER NOT NULL,
            routed_model TEXT NOT NULL,
            provider TEXT NOT NULL,
            input_tokens INTEGER DEFAULT 0,
            output_tokens INTEGER DEFAULT 0,
            cost_usd REAL DEFAULT 0.0,
            latency_ms INTEGER DEFAULT 0,
            quality_score REAL,
            was_escalated INTEGER DEFAULT 0,
            escalated_model TEXT,
            escalation_cost_delta REAL,
            error TEXT
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_timestamp ON requests(timestamp)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_model ON requests(routed_model)")
    conn.commit()
    conn.close()
    logger.info("Database initialized")


def log_request(log: RequestLog) -> int:
    conn = sqlite3.connect(DB_PATH)
    cur = conn.execute("""
        INSERT INTO requests (
            timestamp, prompt_hash, prompt_preview, complexity_tier,
            routed_model, provider, input_tokens, output_tokens,
            cost_usd, latency_ms, quality_score, was_escalated,
            escalated_model, escalation_cost_delta, error
        ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
    """, (
        log.timestamp, log.prompt_hash, log.prompt_preview, log.complexity_tier,
        log.routed_model, log.provider, log.input_tokens, log.output_tokens,
        log.cost_usd, log.latency_ms, log.quality_score, int(log.was_escalated),
        log.escalated_model, log.escalation_cost_delta, log.error,
    ))
    conn.commit()
    row_id = cur.lastrowid
    conn.close()
    return row_id


def update_quality_score(request_id: int, score: float, was_escalated: bool,
                          escalated_model: Optional[str] = None,
                          cost_delta: Optional[float] = None):
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        UPDATE requests
        SET quality_score = ?, was_escalated = ?, escalated_model = ?, escalation_cost_delta = ?
        WHERE id = ?
    """, (score, int(was_escalated), escalated_model, cost_delta, request_id))
    conn.commit()
    conn.close()


def get_stats(days: int = 7) -> dict:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row

    # Total cost
    row = conn.execute("""
        SELECT
            COUNT(*) as total_requests,
            SUM(cost_usd) as total_cost,
            AVG(cost_usd) as avg_cost,
            AVG(latency_ms) as avg_latency,
            AVG(quality_score) as avg_quality,
            SUM(was_escalated) as escalations
        FROM requests
        WHERE timestamp >= datetime('now', ?)
        AND error IS NULL
    """, (f"-{days} days",)).fetchone()

    stats = dict(row) if row else {}

    # Cost by model
    model_rows = conn.execute("""
        SELECT routed_model, COUNT(*) as count, SUM(cost_usd) as cost
        FROM requests
        WHERE timestamp >= datetime('now', ?)
        AND error IS NULL
        GROUP BY routed_model
    """, (f"-{days} days",)).fetchall()
    stats["by_model"] = [dict(r) for r in model_rows]

    # Daily costs
    daily_rows = conn.execute("""
        SELECT DATE(timestamp) as date, SUM(cost_usd) as cost, COUNT(*) as requests
        FROM requests
        WHERE timestamp >= datetime('now', ?)
        AND error IS NULL
        GROUP BY DATE(timestamp)
        ORDER BY date
    """, (f"-{days} days",)).fetchall()
    stats["daily"] = [dict(r) for r in daily_rows]

    # Tier distribution
    tier_rows = conn.execute("""
        SELECT complexity_tier, COUNT(*) as count
        FROM requests
        WHERE timestamp >= datetime('now', ?)
        GROUP BY complexity_tier
    """, (f"-{days} days",)).fetchall()
    stats["by_tier"] = [dict(r) for r in tier_rows]

    conn.close()
    return stats


def get_recent_requests(limit: int = 50) -> list[dict]:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    rows = conn.execute("""
        SELECT * FROM requests ORDER BY timestamp DESC LIMIT ?
    """, (limit,)).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_gpt4o_baseline_cost(days: int = 7) -> float:
    """Calculate what everything would have cost using GPT-4o."""
    from app.models import MODEL_REGISTRY
    gpt4o = MODEL_REGISTRY["gpt-4o"]

    conn = sqlite3.connect(DB_PATH)
    row = conn.execute("""
        SELECT SUM(input_tokens) as total_input, SUM(output_tokens) as total_output
        FROM requests
        WHERE timestamp >= datetime('now', ?)
        AND error IS NULL
    """, (f"-{days} days",)).fetchone()
    conn.close()

    if not row or row[0] is None:
        return 0.0

    return gpt4o.estimate_cost(int(row[0]), int(row[1]))

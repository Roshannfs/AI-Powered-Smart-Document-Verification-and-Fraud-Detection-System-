"""
SQLite Database Module for Storing and Querying Verification Audit Trails.
Tracks document verifications, hashes, scores, and supports duplicate queries and history purge.
"""

import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional

from config.config import DB_PATH
from modules.utils import perceptual_hash_distance


def get_db_connection() -> sqlite3.Connection:
    """Creates a thread-safe connection to the SQLite audit database."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def init_database():
    """Initializes the SQLite database schema if not already present."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS verifications (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT NOT NULL,
            sha256_hash TEXT NOT NULL,
            phash TEXT,
            timestamp TEXT NOT NULL,
            document_type TEXT NOT NULL,
            classification_confidence REAL,
            ocr_confidence REAL,
            risk_score INTEGER NOT NULL,
            risk_level TEXT NOT NULL,
            extracted_fields_json TEXT,
            verification_checks_json TEXT,
            reasons_json TEXT,
            report_path TEXT
        );
        """)
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_sha256 ON verifications(sha256_hash);")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_timestamp ON verifications(timestamp);")
        conn.commit()


# Initialize schema on module load
init_database()


def insert_verification_record(
    filename: str,
    sha256_hash: str,
    phash: str,
    document_type: str,
    classification_confidence: float,
    ocr_confidence: float,
    risk_score: int,
    risk_level: str,
    extracted_fields: Dict[str, Any],
    verification_checks: List[Dict[str, Any]],
    reasons: List[str],
    report_path: Optional[str] = None
) -> int:
    """Inserts an analysis record into the SQLite database and returns the row id."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
        INSERT INTO verifications (
            filename, sha256_hash, phash, timestamp,
            document_type, classification_confidence, ocr_confidence,
            risk_score, risk_level, extracted_fields_json,
            verification_checks_json, reasons_json, report_path
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            filename,
            sha256_hash,
            phash,
            timestamp,
            document_type,
            round(classification_confidence, 2),
            round(ocr_confidence, 2),
            risk_score,
            risk_level,
            json.dumps(extracted_fields),
            json.dumps(verification_checks),
            json.dumps(reasons),
            report_path
        ))
        conn.commit()
        return cursor.lastrowid


def check_for_duplicate(sha256_hash: str, phash: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """
    Checks if an exact SHA-256 hash or close perceptual hash exists in database.
    Returns matched record dict or None.
    """
    with get_db_connection() as conn:
        cursor = conn.cursor()

        # 1. Exact cryptographic match
        cursor.execute("""
        SELECT id, filename, timestamp, document_type, risk_score, risk_level
        FROM verifications WHERE sha256_hash = ?
        ORDER BY id DESC LIMIT 1
        """, (sha256_hash,))
        row = cursor.fetchone()
        if row:
            d = dict(row)
            d["match_type"] = "exact_sha256"
            return d

        # 2. Perceptual visual hash match (Hamming distance <= 4)
        if phash:
            cursor.execute("SELECT id, filename, timestamp, document_type, risk_score, phash FROM verifications WHERE phash IS NOT NULL")
            rows = cursor.fetchall()
            for r in rows:
                candidate_phash = r["phash"]
                dist = perceptual_hash_distance(phash, candidate_phash)
                if dist <= 4:
                    return {
                        "id": r["id"],
                        "filename": r["filename"],
                        "timestamp": r["timestamp"],
                        "document_type": r["document_type"],
                        "risk_score": r["risk_score"],
                        "match_type": f"visual_phash (distance {dist})"
                    }

    return None


def get_all_verifications(limit: int = 100) -> List[Dict[str, Any]]:
    """Retrieves verification audit records sorted by most recent."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
        SELECT id, filename, timestamp, document_type, classification_confidence,
               ocr_confidence, risk_score, risk_level, report_path
        FROM verifications ORDER BY id DESC LIMIT ?
        """, (limit,))
        return [dict(row) for row in cursor.fetchall()]


def get_verification_by_id(record_id: int) -> Optional[Dict[str, Any]]:
    """Retrieves full details of a specific verification record by ID."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM verifications WHERE id = ?", (record_id,))
        row = cursor.fetchone()
        if not row:
            return None
        data = dict(row)
        data["extracted_fields"] = json.loads(data.get("extracted_fields_json") or "{}")
        data["verification_checks"] = json.loads(data.get("verification_checks_json") or "[]")
        data["reasons"] = json.loads(data.get("reasons_json") or "[]")
        return data


def delete_verification(record_id: int) -> bool:
    """Deletes a single verification record."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM verifications WHERE id = ?", (record_id,))
        conn.commit()
        return cursor.rowcount > 0


def clear_all_history() -> int:
    """Clears all audit trail history."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM verifications")
        conn.commit()
        return cursor.rowcount

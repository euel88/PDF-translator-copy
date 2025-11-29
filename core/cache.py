"""
Translation cache module.
"""
import os
import json
import hashlib
import sqlite3
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Dict, Any
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    """A cached translation entry."""
    key: str
    source_text: str
    translated_text: str
    source_lang: str
    target_lang: str
    service: str
    created_at: datetime
    accessed_at: datetime
    hit_count: int = 0


class TranslationCache:
    """
    SQLite-based translation cache.

    Caches translation results to avoid redundant API calls
    and speed up repeated translations.
    """

    def __init__(self, cache_dir: str = ".cache/translations", max_age_days: int = 30):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = self.cache_dir / "translations.db"
        self.max_age_days = max_age_days
        self._init_db()

    def _init_db(self):
        """Initialize the SQLite database."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS translations (
                    key TEXT PRIMARY KEY,
                    source_text TEXT NOT NULL,
                    translated_text TEXT NOT NULL,
                    source_lang TEXT NOT NULL,
                    target_lang TEXT NOT NULL,
                    service TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    accessed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    hit_count INTEGER DEFAULT 0
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_langs
                ON translations(source_lang, target_lang, service)
            """)
            conn.commit()

    def _generate_key(
        self,
        text: str,
        source_lang: str,
        target_lang: str,
        service: str
    ) -> str:
        """Generate a unique cache key."""
        content = f"{text}|{source_lang}|{target_lang}|{service}"
        return hashlib.sha256(content.encode()).hexdigest()

    def get(
        self,
        text: str,
        source_lang: str,
        target_lang: str,
        service: str
    ) -> Optional[str]:
        """
        Get a cached translation.

        Args:
            text: Source text
            source_lang: Source language code
            target_lang: Target language code
            service: Translation service

        Returns:
            Cached translation or None if not found
        """
        key = self._generate_key(text, source_lang, target_lang, service)

        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute(
                    """
                    SELECT translated_text, created_at FROM translations
                    WHERE key = ?
                    """,
                    (key,)
                )
                row = cursor.fetchone()

                if row:
                    translated_text, created_at = row

                    # Check if entry is expired
                    created = datetime.fromisoformat(created_at)
                    if datetime.now() - created > timedelta(days=self.max_age_days):
                        # Remove expired entry
                        conn.execute("DELETE FROM translations WHERE key = ?", (key,))
                        conn.commit()
                        return None

                    # Update access time and hit count
                    conn.execute(
                        """
                        UPDATE translations
                        SET accessed_at = CURRENT_TIMESTAMP, hit_count = hit_count + 1
                        WHERE key = ?
                        """,
                        (key,)
                    )
                    conn.commit()

                    return translated_text

        except Exception as e:
            logger.warning(f"Cache get error: {e}")

        return None

    def set(
        self,
        text: str,
        translated_text: str,
        source_lang: str,
        target_lang: str,
        service: str
    ):
        """
        Store a translation in cache.

        Args:
            text: Source text
            translated_text: Translated text
            source_lang: Source language code
            target_lang: Target language code
            service: Translation service
        """
        key = self._generate_key(text, source_lang, target_lang, service)

        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO translations
                    (key, source_text, translated_text, source_lang, target_lang, service)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (key, text, translated_text, source_lang, target_lang, service)
                )
                conn.commit()

        except Exception as e:
            logger.warning(f"Cache set error: {e}")

    def clear(self, older_than_days: Optional[int] = None):
        """
        Clear cache entries.

        Args:
            older_than_days: If specified, only clear entries older than this
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                if older_than_days is not None:
                    cutoff = datetime.now() - timedelta(days=older_than_days)
                    conn.execute(
                        "DELETE FROM translations WHERE created_at < ?",
                        (cutoff.isoformat(),)
                    )
                else:
                    conn.execute("DELETE FROM translations")
                conn.commit()

        except Exception as e:
            logger.warning(f"Cache clear error: {e}")

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("SELECT COUNT(*) FROM translations")
                total_entries = cursor.fetchone()[0]

                cursor = conn.execute("SELECT SUM(hit_count) FROM translations")
                total_hits = cursor.fetchone()[0] or 0

                cursor = conn.execute(
                    "SELECT service, COUNT(*) FROM translations GROUP BY service"
                )
                by_service = dict(cursor.fetchall())

                return {
                    "total_entries": total_entries,
                    "total_hits": total_hits,
                    "by_service": by_service,
                    "db_size_mb": round(os.path.getsize(self.db_path) / (1024 * 1024), 2),
                }

        except Exception as e:
            logger.warning(f"Cache stats error: {e}")
            return {"error": str(e)}

    def delete(
        self,
        text: str,
        source_lang: str,
        target_lang: str,
        service: str
    ) -> bool:
        """Delete a specific cache entry."""
        key = self._generate_key(text, source_lang, target_lang, service)

        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("DELETE FROM translations WHERE key = ?", (key,))
                conn.commit()
                return cursor.rowcount > 0

        except Exception as e:
            logger.warning(f"Cache delete error: {e}")
            return False


# Global cache instance
_cache: Optional[TranslationCache] = None


def get_cache(cache_dir: str = ".cache/translations") -> TranslationCache:
    """Get the global cache instance."""
    global _cache
    if _cache is None:
        _cache = TranslationCache(cache_dir)
    return _cache

"""
Cache module for vLLM profile_run results on Strix Halo.

On first boot, vLLM runs ~7 min of synthetic forward passes to size the KV
cache pool. This module fingerprints the config and caches the result so
subsequent restarts (same config) skip profiling and boot in ~95 s.

Usage: imported at runtime by Patch 16 in scripts/patch_strix.py.
       Must be placed on sys.path (e.g. /opt/vllm_profile_cache.py).
"""
import hashlib
import json
import logging
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

# Config parameters that affect KV cache memory sizing.
# If any of these change, the cached result is invalid.
_FINGERPRINT_KEYS = [
    ("model_config", "model"),
    ("cache_config", "gpu_memory_utilization"),
    ("model_config", "max_model_len"),
    ("scheduler_config", "max_num_seqs"),
    ("cache_config", "block_size"),
    ("cache_config", "cache_dtype"),
    ("speculative_config", "method"),
    ("speculative_config", "model"),
]


def make_fingerprint(vllm_config) -> str:
    """Build a stable SHA256 fingerprint of the vLLM config."""
    fp = {}
    for section, key in _FINGERPRINT_KEYS:
        section_obj = getattr(vllm_config, section, None)
        if section_obj is not None:
            fp[key] = getattr(section_obj, key, None)
        else:
            fp[key] = None
    raw = json.dumps(fp, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def _cache_file_path(cache_dir: str, fingerprint: str) -> Path:
    return Path(cache_dir) / f"profile_cache_{fingerprint}.json"


def read_cached_kv_cache_memory_bytes(cache_dir: str, vllm_config) -> int | None:
    """Read cached KV cache memory from a previous profile_run.

    Returns the cached value (int bytes) or None if cache miss/invalid.
    """
    try:
        fingerprint = make_fingerprint(vllm_config)
        path = _cache_file_path(cache_dir, fingerprint)
        if not path.exists():
            return None
        data = json.loads(path.read_text())
        if data.get("fingerprint") != fingerprint:
            logger.debug("Profile cache fingerprint mismatch")
            return None
        value = data.get("available_kv_cache_memory_bytes")
        if value is not None:
            ts = data.get("timestamp", "unknown")
            logger.info(
                "Profile cache hit: %d bytes (cached %s)", value, ts
            )
            return int(value)
    except Exception:
        logger.debug("Profile cache read failed; falling back to profiling")
    return None


def write_cached_kv_cache_memory_bytes(
    cache_dir: str, value_bytes: int, vllm_config
) -> None:
    """Write the profile_run result to a cache file atomically."""
    try:
        os.makedirs(cache_dir, exist_ok=True)
        fingerprint = make_fingerprint(vllm_config)
        path = _cache_file_path(cache_dir, fingerprint)
        data = {
            "fingerprint": fingerprint,
            "available_kv_cache_memory_bytes": value_bytes,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        # Atomic write: write to temp file then rename.
        fd, tmp_path = tempfile.mkstemp(dir=cache_dir, suffix=".tmp")
        try:
            with os.fdopen(fd, "w") as f:
                json.dump(data, f, indent=2)
                f.write("\n")
            os.replace(tmp_path, path)
            logger.info("Cached KV cache memory: %d bytes", value_bytes)
        except Exception:
            os.unlink(tmp_path)
            raise
    except Exception:
        logger.debug("Profile cache write failed (non-fatal)")

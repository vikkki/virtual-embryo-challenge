"""Thin client for the Virtual Embryo API (https://api.virtualembryo.ai).

Safety: official uploads are irreversible in P3 (2 per test board for the whole phase, and the first
accepted one permanently ends P2 team-wide). `format_check` is free; `submit` refuses to send
`confirm_final` unless the caller passes it explicitly.
"""
from __future__ import annotations

import os
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[2]


def _env():
    try:
        from dotenv import load_dotenv

        load_dotenv(ROOT / ".env")
    except ImportError:  # minimal fallback: KEY=VALUE lines
        envf = ROOT / ".env"
        if envf.exists():
            for line in envf.read_text().splitlines():
                k, sep, v = line.strip().partition("=")
                if sep and not k.startswith("#") and v:
                    os.environ.setdefault(k.strip(), v.strip().strip("'\""))
    base = os.environ.get("VE_API_BASE", "https://api.virtualembryo.ai").rstrip("/")
    return base, os.environ.get("VE_CHALLENGE_API_KEY", "")


def _req(method: str, path: str, *, auth: bool = True, **kw) -> dict:
    base, key = _env()
    headers = kw.pop("headers", {})
    if auth:
        if not key:
            raise RuntimeError("VE_CHALLENGE_API_KEY not set (put it in .env)")
        headers["Authorization"] = f"Bearer {key}"
    for attempt in range(4):
        r = requests.request(method, f"{base}{path}", headers=headers, timeout=600, **kw)
        if r.status_code < 500:
            break
        time.sleep(2 ** (attempt + 1))
    if r.status_code >= 400:
        raise RuntimeError(f"{method} {path} -> {r.status_code}: {r.text[:2000]}")
    return r.json()


# public
def phase():
    return _req("GET", "/challenge/phase", auth=False)


def boards():
    return _req("GET", "/challenge/boards", auth=False)


# team
def me():
    return _req("GET", "/challenge/me")


def quota():
    return _req("GET", "/challenge/integration/quota")


def manifest():
    return _req("GET", "/challenge/data/manifest")


def download(key: str, dest_root: Path = ROOT / "data", overwrite: bool = False) -> Path:
    dest = dest_root / key
    if dest.exists() and not overwrite:
        return dest
    url = _req("GET", "/challenge/data/link", params={"key": key})["url"]
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    with requests.get(url, stream=True, timeout=600) as r:  # signed URL: no API key
        r.raise_for_status()
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(1 << 20):
                f.write(chunk)
    tmp.rename(dest)
    return dest


def _fields(task: str, setting: str | None, mode: str | None, split: str | None) -> dict:
    d = {"task": task}
    if setting:
        d["setting"] = setting
    if mode:
        d["mode"] = mode
    if split:
        d["split"] = split
    return d


def format_check(path: str | Path, task: str, setting=None, mode=None, split=None) -> dict:
    """Free: no submission created, no attempt spent, no P2 lock."""
    data = _fields(task, setting, mode, split) | {"format_only": "true"}
    with open(path, "rb") as f:
        return _req("POST", "/challenge/submissions", data=data, files={"file": f})


def submit(path, task, *, idempotency_key: str, setting=None, mode=None, split=None,
           model=None, confirm_final: bool = False, **extra) -> dict:
    """Official upload. Spends quota. Never pass confirm_final=True without a human decision."""
    data = _fields(task, setting, mode, split)
    if model:
        data["model"] = model
    data.update({k: v for k, v in extra.items() if v is not None})
    if confirm_final:
        data["confirm_final"] = "true"
    with open(path, "rb") as f:
        return _req("POST", "/challenge/submissions", data=data, files={"file": f},
                    headers={"Idempotency-Key": idempotency_key})


def submission(sub_id: str) -> dict:
    return _req("GET", f"/challenge/submissions/{sub_id}")


def submissions() -> dict:
    return _req("GET", "/challenge/submissions")

import json
from datetime import datetime, timezone
import redis, os
from dotenv import load_dotenv
load_dotenv()

REDIS_URL = os.getenv("REDIS_URL", "redis://127.0.0.1:6379/0")

def rdb():
    return redis.from_url(REDIS_URL)

def now_iso():
    return datetime.now(timezone.utc).isoformat()

def save_token_state(token: str, data: dict):
    rdb().set(f"token:{token}", json.dumps(data), ex=3600)  # expire 1h

def load_token_state(token: str):
    raw = rdb().get(f"token:{token}")
    return json.loads(raw) if raw else None

def update_token_field(token: str, field: str, value):
    data = load_token_state(token) or {}
    data[field] = value
    save_token_state(token, data)

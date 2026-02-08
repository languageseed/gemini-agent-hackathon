"""
Session Persistence

Simple Redis-backed session storage for marathon agent resume capability.
Total: ~50 lines
"""

import os
import json
import uuid
from dataclasses import dataclass, field
from typing import Optional, Any
from datetime import datetime

import structlog

logger = structlog.get_logger()


@dataclass
class Session:
    """Agent session state."""
    id: Optional[str] = None
    messages: list = field(default_factory=list)  # Gemini Content objects
    iteration: int = 0
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())
    metadata: dict = field(default_factory=dict)
    
    def __post_init__(self):
        if self.id is None:
            self.id = str(uuid.uuid4())


class SessionStore:
    """
    Redis-backed session storage with access control.
    
    Sessions are scoped by owner_id (derived from API key hash) to prevent
    unauthorized access. Falls back to in-memory storage if Redis is not available.
    """
    
    def __init__(self, redis_url: Optional[str] = None):
        self.redis_url = redis_url or os.environ.get("REDIS_URL")
        self._redis = None
        self._memory_store: dict[str, dict] = {}  # Fallback
    
    async def _get_redis(self):
        """Get or create Redis connection."""
        if self._redis is None and self.redis_url:
            try:
                import redis.asyncio as redis
                self._redis = redis.from_url(self.redis_url)
                await self._redis.ping()
                logger.info("redis_connected")
            except Exception as e:
                logger.warning("redis_connection_failed", error=str(e), fallback="memory")
                self._redis = None
        return self._redis
    
    @staticmethod
    def _hash_owner(api_key: str) -> str:
        """Hash API key to create an owner scope (never store raw keys)."""
        import hashlib
        return hashlib.sha256(api_key.encode()).hexdigest()[:16]
    
    def _session_key(self, session_id: str, owner_id: str = "") -> str:
        """Build storage key scoped to owner."""
        if owner_id:
            return f"session:{owner_id}:{session_id}"
        return f"session:{session_id}"
    
    async def save(self, session: Session, owner_id: str = "") -> None:
        """Save session state, scoped to owner."""
        session.updated_at = datetime.now().isoformat()
        
        # Serialize messages (Content objects) to JSON-compatible format
        data = {
            "id": session.id,
            "owner_id": owner_id,
            "iteration": session.iteration,
            "created_at": session.created_at,
            "updated_at": session.updated_at,
            "metadata": session.metadata,
            "messages": self._serialize_messages(session.messages)
        }
        
        key = self._session_key(session.id, owner_id)
        
        redis = await self._get_redis()
        if redis:
            try:
                await redis.set(key, json.dumps(data), ex=86400 * 7)  # 7 days TTL
                return
            except Exception as e:
                logger.warning("redis_save_failed", error=str(e))
        
        # Fallback to memory
        self._memory_store[key] = data
    
    async def load(self, session_id: str, owner_id: str = "") -> Optional[Session]:
        """Load session state. Must match owner_id used during save."""
        key = self._session_key(session_id, owner_id)
        redis = await self._get_redis()
        data = None
        
        if redis:
            try:
                raw = await redis.get(key)
                if raw:
                    data = json.loads(raw)
            except Exception as e:
                logger.warning("redis_load_failed", error=str(e))
        
        # Fallback to memory
        if data is None:
            data = self._memory_store.get(key)
        
        if data is None:
            return None
        
        # Verify owner matches (defense in depth)
        if owner_id and data.get("owner_id") and data["owner_id"] != owner_id:
            logger.warning("session_access_denied", session_id=session_id)
            return None
        
        return Session(
            id=data["id"],
            messages=self._deserialize_messages(data.get("messages", [])),
            iteration=data.get("iteration", 0),
            created_at=data.get("created_at", ""),
            updated_at=data.get("updated_at", ""),
            metadata=data.get("metadata", {})
        )
    
    async def delete(self, session_id: str, owner_id: str = "") -> None:
        """Delete session. Must match owner_id."""
        key = self._session_key(session_id, owner_id)
        redis = await self._get_redis()
        if redis:
            try:
                await redis.delete(key)
            except Exception as e:
                logger.warning("redis_delete_failed", error=str(e))
        
        self._memory_store.pop(key, None)
    
    async def list_sessions(self, owner_id: str = "", limit: int = 100) -> list[str]:
        """List recent session IDs for an owner."""
        redis = await self._get_redis()
        prefix = f"session:{owner_id}:" if owner_id else "session:"
        
        if redis:
            try:
                keys = await redis.keys(f"{prefix}*")
                return [k.decode().split(":")[-1] for k in keys[:limit]]
            except Exception as e:
                logger.warning("redis_list_failed", error=str(e))
        
        return [
            k.split(":")[-1] for k in self._memory_store.keys()
            if k.startswith(prefix)
        ][:limit]
    
    def _serialize_messages(self, messages: list) -> list[dict]:
        """Convert Gemini Content objects to JSON-serializable dicts."""
        serialized = []
        for msg in messages:
            if hasattr(msg, 'role') and hasattr(msg, 'parts'):
                # Gemini Content object
                parts = []
                for part in msg.parts:
                    if hasattr(part, 'text') and part.text:
                        parts.append({"text": part.text})
                    elif hasattr(part, 'function_call') and part.function_call:
                        fc = part.function_call
                        parts.append({
                            "function_call": {
                                "name": fc.name,
                                "args": dict(fc.args) if fc.args else {}
                            }
                        })
                    elif hasattr(part, 'function_response') and part.function_response:
                        parts.append({"function_response": part.function_response})
                
                serialized.append({
                    "role": msg.role,
                    "parts": parts
                })
            elif isinstance(msg, dict):
                serialized.append(msg)
        
        return serialized
    
    def _deserialize_messages(self, data: list[dict]) -> list:
        """Convert JSON dicts back to Gemini Content objects."""
        from google.genai.types import Content, Part
        
        messages = []
        for msg_data in data:
            parts = []
            for part_data in msg_data.get("parts", []):
                if "text" in part_data:
                    parts.append(Part(text=part_data["text"]))
                elif "function_call" in part_data:
                    # Note: function_call parts may need special handling
                    parts.append(Part(text=f"[Function call: {part_data['function_call']}]"))
                elif "function_response" in part_data:
                    parts.append(Part(function_response=part_data["function_response"]))
            
            messages.append(Content(
                role=msg_data.get("role", "user"),
                parts=parts
            ))
        
        return messages

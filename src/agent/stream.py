"""
SSE Streaming Events

Server-Sent Events for real-time progress updates.
Total: ~40 lines
"""

import json
from dataclasses import dataclass, asdict
from typing import Any, Optional, AsyncIterator
from enum import Enum


class EventType(str, Enum):
    """SSE event types."""
    START = "start"           # Agent started
    THINKING = "thinking"     # Thinking level selected
    TOKEN = "token"           # Text token generated
    TOOL_START = "tool_start" # Tool execution started
    TOOL_RESULT = "tool_result" # Tool execution completed
    CHECKPOINT = "checkpoint" # Session checkpointed
    ERROR = "error"           # Error occurred
    DONE = "done"             # Agent completed


@dataclass
class StreamEvent:
    """A streaming event."""
    type: EventType
    data: dict
    
    def to_sse(self) -> str:
        """Format as Server-Sent Event."""
        event_data = {
            "type": self.type.value,
            **self.data
        }
        return f"data: {json.dumps(event_data)}\n\n"


class EventCollector:
    """Collects events for streaming response."""
    
    def __init__(self):
        self.events: list[StreamEvent] = []
    
    def __call__(self, event: StreamEvent) -> None:
        """Add event to collection."""
        self.events.append(event)
    
    def clear(self) -> None:
        """Clear collected events."""
        self.events = []


async def stream_agent_events(
    events: list[StreamEvent]
) -> AsyncIterator[str]:
    """
    Stream events as SSE.
    
    Usage:
        return StreamingResponse(
            stream_agent_events(events),
            media_type="text/event-stream"
        )
    """
    for event in events:
        yield event.to_sse()


def create_sse_response(events: list[StreamEvent]) -> str:
    """Create complete SSE response body."""
    return "".join(event.to_sse() for event in events)

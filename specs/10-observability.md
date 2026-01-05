# Observability Specification

## Overview

You can't debug what you can't see. This spec defines health endpoints, structured logging, metrics, and debugging tools that make the system transparent and debuggable.

## Design Principles

1. **Everything is observable** - Every component exposes state
2. **Structured logging** - JSON logs, not printf
3. **Health at a glance** - Single endpoint shows system state
4. **Debug modes** - Extra verbosity when needed
5. **Low overhead** - Observability shouldn't slow the system

## Health Endpoint

### `/health` API

Single endpoint returns full system status:

```python
from fastapi import FastAPI
from datetime import datetime

app = FastAPI()

@app.get("/health")
async def health() -> dict:
    """Return comprehensive system health."""
    return {
        "status": "healthy",  # healthy | degraded | unhealthy
        "timestamp": datetime.utcnow().isoformat(),
        "uptime_seconds": get_uptime(),
        "components": {
            "robot": await get_robot_health(),
            "memory": await get_memory_health(),
            "voice": await get_voice_health(),
            "motion": await get_motion_health(),
            "claude": await get_claude_health()
        }
    }
```

### Component Health

```python
async def get_robot_health() -> dict:
    """Robot subsystem health."""
    return {
        "status": "healthy",
        "connected": True,
        "latency_ms": 3.2,
        "is_awake": True,
        "battery_percent": 87.5,
        "errors_last_hour": 0
    }

async def get_memory_health() -> dict:
    """Memory subsystem health."""
    return {
        "status": "healthy",
        "connected": True,
        "memory_count": 1234,
        "context_window_size": 5,
        "last_cleanup": "2025-01-05T10:00:00Z"
    }

async def get_voice_health() -> dict:
    """Voice subsystem health."""
    return {
        "status": "healthy",
        "state": "listening_wake",  # Current pipeline state
        "active_persona": "motoko",
        "wake_word_active": True,
        "openai_connected": True,
        "audio_devices": {
            "input": "USB Microphone Array",
            "output": "Built-in Speaker"
        }
    }

async def get_motion_health() -> dict:
    """Motion subsystem health."""
    return {
        "status": "healthy",
        "tick_rate_hz": 30,
        "active_primary": "idle",
        "active_overlays": [],
        "last_command_ms": 12
    }

async def get_claude_health() -> dict:
    """Claude API health."""
    return {
        "status": "healthy",
        "model": "claude-haiku-4-5-20251001",
        "last_call_ms": 234,
        "calls_last_hour": 42,
        "errors_last_hour": 0
    }
```

### Health Status Logic

```python
def compute_overall_status(components: dict) -> str:
    """Compute overall health from component statuses."""
    statuses = [c["status"] for c in components.values()]

    if all(s == "healthy" for s in statuses):
        return "healthy"
    elif any(s == "unhealthy" for s in statuses):
        return "unhealthy"
    else:
        return "degraded"
```

### Example Response

```json
{
  "status": "healthy",
  "timestamp": "2025-01-05T12:34:56Z",
  "uptime_seconds": 3600,
  "components": {
    "robot": {
      "status": "healthy",
      "connected": true,
      "latency_ms": 3.2,
      "is_awake": true,
      "battery_percent": 87.5
    },
    "memory": {
      "status": "healthy",
      "connected": true,
      "memory_count": 1234
    },
    "voice": {
      "status": "healthy",
      "state": "listening_wake",
      "active_persona": "motoko"
    },
    "motion": {
      "status": "healthy",
      "tick_rate_hz": 30,
      "active_primary": "idle"
    },
    "claude": {
      "status": "healthy",
      "model": "claude-haiku-4-5-20251001",
      "last_call_ms": 234
    }
  }
}
```

## Structured Logging

### Log Format

All logs are structured JSON (via structlog):

```python
import structlog

log = structlog.get_logger()

# Good - structured
log.info("tool_executed", tool="move_head", duration_ms=45, success=True)

# Bad - unstructured
print(f"Executed move_head in 45ms")
```

### Log Levels

| Level | Use |
|-------|-----|
| `DEBUG` | Detailed tracing (off by default) |
| `INFO` | Normal operations |
| `WARNING` | Recoverable issues |
| `ERROR` | Failures requiring attention |

### Standard Log Fields

Every log entry includes:

```python
{
    "timestamp": "2025-01-05T12:34:56.789Z",
    "level": "info",
    "event": "tool_executed",  # What happened
    "component": "robot",       # Which subsystem

    # Event-specific fields
    "tool": "move_head",
    "duration_ms": 45,
    "success": true
}
```

### Component Logging

```python
# Robot
log.info("robot_connected", latency_ms=3.2)
log.info("robot_command", tool="move_head", args={"pitch": 10})
log.error("robot_error", error="Connection lost")

# Voice
log.info("voice_event", event="wake_detected", persona="motoko", confidence=0.87)
log.info("voice_event", event="transcribed", text="hello", duration_ms=450)

# Memory
log.info("memory_stored", type="fact", content_length=50)
log.info("memory_searched", query="user preferences", results=3)

# Motion
log.info("motion_source_changed", primary="emotion", previous="idle")
log.debug("motion_tick", pose={"pitch": 5.2, "yaw": -3.1})

# Claude
log.info("claude_request", model="claude-haiku-4-5", tokens_in=500)
log.info("claude_response", tokens_out=150, duration_ms=890)

# Permissions
log.info("permission_evaluated", tool="store_memory", tier="confirm", allowed=True)
```

## Debug Modes

### Voice Debug Mode

```bash
python -m reachy_agent run --voice --debug-voice
```

Outputs full event trace:

```
[12:34:56.789] voice_event event=wake_detected persona=motoko confidence=0.87
[12:34:56.792] voice_event event=listening_start persona=motoko
[12:34:59.123] voice_event event=listening_end audio_duration=2.331
[12:34:59.456] voice_event event=transcribed text="What time is it?" confidence=0.94
[12:34:59.458] voice_event event=processing text="What time is it?"
[12:35:00.234] voice_event event=response text="I don't have access..."
[12:35:00.236] voice_event event=speaking_start text="I don't have..." voice=nova
[12:35:02.567] voice_event event=speaking_end
```

### Motion Debug Mode

```bash
python -m reachy_agent run --debug-motion
```

Outputs tick-level data (30Hz - lots of output):

```
[12:34:56.000] motion_tick primary=idle overlay=[] pose={pitch: 2.1, yaw: -5.3}
[12:34:56.033] motion_tick primary=idle overlay=[] pose={pitch: 2.3, yaw: -5.1}
...
```

### Claude Debug Mode

```bash
python -m reachy_agent run --debug-claude
```

Shows full API payloads:

```
[12:34:56.000] claude_request model=claude-haiku-4-5 messages=[...]
[12:34:56.890] claude_response content=[...] stop_reason=end_turn
```

## Metrics

### Prometheus-Style Metrics

```python
from dataclasses import dataclass, field
from collections import defaultdict
import time

@dataclass
class Metrics:
    """Simple metrics collector."""

    # Counters
    tool_calls: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    errors: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    voice_events: dict[str, int] = field(default_factory=lambda: defaultdict(int))

    # Gauges
    memory_count: int = 0
    context_window_size: int = 0
    battery_percent: float = 0

    # Histograms (simplified)
    claude_latencies: list[float] = field(default_factory=list)
    robot_latencies: list[float] = field(default_factory=list)

    def record_tool_call(self, tool: str, success: bool, duration_ms: float):
        self.tool_calls[tool] += 1
        if not success:
            self.errors[tool] += 1
        self.robot_latencies.append(duration_ms)

    def record_claude_call(self, duration_ms: float):
        self.claude_latencies.append(duration_ms)

    def record_voice_event(self, event: str):
        self.voice_events[event] += 1

    def to_dict(self) -> dict:
        """Export metrics as dict."""
        return {
            "tool_calls": dict(self.tool_calls),
            "errors": dict(self.errors),
            "voice_events": dict(self.voice_events),
            "memory_count": self.memory_count,
            "battery_percent": self.battery_percent,
            "latency": {
                "claude_p50_ms": percentile(self.claude_latencies, 50),
                "claude_p99_ms": percentile(self.claude_latencies, 99),
                "robot_p50_ms": percentile(self.robot_latencies, 50),
                "robot_p99_ms": percentile(self.robot_latencies, 99)
            }
        }
```

### `/metrics` Endpoint

```python
@app.get("/metrics")
async def metrics() -> dict:
    """Return current metrics."""
    return global_metrics.to_dict()
```

Example response:

```json
{
  "tool_calls": {
    "move_head": 142,
    "speak": 45,
    "get_status": 89
  },
  "errors": {
    "move_head": 2
  },
  "voice_events": {
    "wake_detected": 23,
    "transcribed": 22,
    "interrupted": 1
  },
  "memory_count": 1234,
  "battery_percent": 87.5,
  "latency": {
    "claude_p50_ms": 234,
    "claude_p99_ms": 890,
    "robot_p50_ms": 12,
    "robot_p99_ms": 45
  }
}
```

## Audit Logging

Permission decisions logged to file for review:

```python
# ~/.reachy/audit.jsonl
{"timestamp": "2025-01-05T10:30:00Z", "tool": "move_head", "tier": "autonomous", "allowed": true}
{"timestamp": "2025-01-05T10:30:15Z", "tool": "store_memory", "tier": "confirm", "allowed": true, "user_response": "approved"}
{"timestamp": "2025-01-05T10:31:00Z", "tool": "exec_command", "tier": "forbidden", "allowed": false}
```

## CLI Health Check

```bash
$ python -m reachy_agent check

🤖 Reachy Agent Health Check

Robot:    ✅ Connected (3.2ms latency, 87% battery)
Memory:   ✅ Connected (1234 memories)
Voice:    ✅ Ready (listening for wake word)
Motion:   ✅ Running (30Hz, idle active)
Claude:   ✅ Available (haiku-4-5)

Overall:  ✅ Healthy
```

With issues:

```bash
$ python -m reachy_agent check

🤖 Reachy Agent Health Check

Robot:    ❌ Disconnected (Connection refused)
Memory:   ✅ Connected (1234 memories)
Voice:    ⚠️ Degraded (microphone not detected)
Motion:   ❌ Stopped (no robot connection)
Claude:   ✅ Available (haiku-4-5)

Overall:  ❌ Unhealthy

Issues:
  - Robot connection failed: Connection refused to localhost:8000
  - Microphone 'USB Microphone' not found
```

## Log Rotation

```yaml
# config/default.yaml
logging:
  level: info
  format: json
  file: ~/.reachy/logs/agent.log
  rotation:
    max_size_mb: 10
    max_files: 5
```

## Configuration

```yaml
# config/default.yaml
observability:
  health_port: 8080
  metrics_enabled: true
  audit_enabled: true
  audit_path: ~/.reachy/audit.jsonl

logging:
  level: info
  format: json  # json | pretty
  debug_voice: false
  debug_motion: false
  debug_claude: false
```

## Related Specs

- [08-agent-loop.md](./08-agent-loop.md) - Main coordinator (exposes health)
- [06-permissions.md](./06-permissions.md) - Permission audit logging
- [03-voice.md](./03-voice.md) - Voice debug events

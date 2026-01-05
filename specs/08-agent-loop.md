# Agent Loop Specification

## Overview

The Agent Loop is the brain - it coordinates Claude, MCP tools, voice, motion, memory, and permissions into a coherent whole. It's the main entry point that orchestrates everything else.

## Design Principles

1. **Single coordinator** - One class ties everything together
2. **Async-first** - Everything is async, no blocking
3. **Mode-agnostic** - Same core works for text, voice, or API modes
4. **Graceful degradation** - Missing components don't crash the system
5. **Observable** - Every step can be monitored

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      ReachyAgentLoop                             │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │                    Core Loop                              │   │
│  │         Input → Context → Claude → Tools → Output         │   │
│  └──────────────────────────────────────────────────────────┘   │
│                              │                                   │
│      ┌───────────────────────┼───────────────────────────────┐  │
│      │           │           │           │           │       │  │
│      ▼           ▼           ▼           ▼           ▼       │  │
│  ┌───────┐  ┌───────┐  ┌───────┐  ┌───────┐  ┌───────┐      │  │
│  │Claude │  │ Robot │  │Memory │  │ Voice │  │Motion │      │  │
│  │  API  │  │  MCP  │  │  MCP  │  │Pipeline│ │Control│      │  │
│  └───────┘  └───────┘  └───────┘  └───────┘  └───────┘      │  │
│                              │                                   │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │                   Permission Hooks                        │   │
│  │              Pre-tool validation & audit                  │   │
│  └──────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

## Core Implementation

```python
import anthropic
from dataclasses import dataclass, field
from typing import AsyncIterator
import structlog

@dataclass
class AgentConfig:
    """Configuration for the agent loop."""
    model: str = "claude-haiku-4-5-20251001"
    name: str = "Jarvis"
    max_tokens: int = 4096
    temperature: float = 0.7

    # Component flags
    enable_voice: bool = False
    enable_motion: bool = True
    mock_hardware: bool = False

    # Paths
    system_prompt_path: str = "prompts/system.md"
    persona_path: str | None = None

@dataclass
class ConversationTurn:
    """A single turn in conversation."""
    role: str  # "user" or "assistant"
    content: str
    tool_calls: list[dict] = field(default_factory=list)
    tool_results: list[dict] = field(default_factory=list)

class ReachyAgentLoop:
    """Main agent coordinator."""

    def __init__(self, config: AgentConfig):
        self._config = config
        self._log = structlog.get_logger()

        # Claude client
        self._client = anthropic.AsyncAnthropic()

        # Components (initialized in start())
        self._robot: ReachyClient | None = None
        self._memory: MemoryManager | None = None
        self._voice: VoicePipeline | None = None
        self._motion: BlendController | None = None
        self._permissions: PermissionEvaluator | None = None

        # MCP tools
        self._tools: list[dict] = []

        # State
        self._system_prompt: str = ""
        self._running = False

    # =========== Lifecycle ===========

    async def start(self):
        """Initialize all components and start the agent."""
        self._log.info("agent_starting", config=self._config)

        # Load system prompt
        self._system_prompt = self._load_system_prompt()

        # Initialize robot client
        if self._config.mock_hardware:
            self._robot = MockClient()
        else:
            self._robot = SDKClient()
        await self._robot.connect()

        # Initialize memory
        self._memory = MemoryManager()

        # Initialize permissions
        self._permissions = load_permissions()

        # Initialize motion controller
        if self._config.enable_motion:
            self._motion = BlendController(self._robot)
            await self._motion.start()

        # Initialize voice pipeline
        if self._config.enable_voice:
            self._voice = VoicePipeline(
                agent=self,
                persona_manager=PersonaManager(),
                debug=False
            )
            await self._voice.start()

        # Build tool list from MCP servers
        self._tools = self._build_tool_list()

        self._running = True
        self._log.info("agent_started")

    async def stop(self):
        """Gracefully shutdown all components."""
        self._log.info("agent_stopping")
        self._running = False

        if self._voice:
            await self._voice.stop()

        if self._motion:
            await self._motion.stop()

        if self._robot:
            await self._robot.disconnect()

        self._log.info("agent_stopped")

    # =========== Main Processing ===========

    async def process(self, user_input: str) -> str:
        """Process a single user input and return response."""
        self._log.info("processing_input", input=user_input[:100])

        # Add to memory context window
        self._memory.add_to_context_window("user", user_input)

        # Build messages with context
        messages = self._build_messages(user_input)

        # Call Claude with tools
        response = await self._call_claude(messages)

        # Process tool calls if any
        while response.stop_reason == "tool_use":
            # Extract tool calls
            tool_calls = [
                block for block in response.content
                if block.type == "tool_use"
            ]

            # Execute tools (with permission checks)
            tool_results = await self._execute_tools(tool_calls)

            # Continue conversation with results
            messages.append({"role": "assistant", "content": response.content})
            messages.append({"role": "user", "content": tool_results})

            response = await self._call_claude(messages)

        # Extract final text response
        assistant_text = self._extract_text(response)

        # Add to memory context window
        self._memory.add_to_context_window("assistant", assistant_text)

        self._log.info("processing_complete", response_length=len(assistant_text))
        return assistant_text

    async def _call_claude(self, messages: list[dict]) -> anthropic.types.Message:
        """Call Claude API with messages and tools."""
        return await self._client.messages.create(
            model=self._config.model,
            max_tokens=self._config.max_tokens,
            temperature=self._config.temperature,
            system=self._system_prompt,
            messages=messages,
            tools=self._tools
        )

    async def _execute_tools(self, tool_calls: list) -> list[dict]:
        """Execute tool calls with permission checks."""
        results = []

        for tool_call in tool_calls:
            tool_name = tool_call.name
            tool_input = tool_call.input

            # Check permission
            decision = self._permissions.evaluate(tool_name, tool_input)

            if decision.tier == PermissionTier.FORBIDDEN:
                results.append({
                    "type": "tool_result",
                    "tool_use_id": tool_call.id,
                    "content": f"Error: {decision.reason}",
                    "is_error": True
                })
                continue

            if decision.tier == PermissionTier.CONFIRM:
                # Ask user for confirmation
                approved = await self._confirm_tool(tool_name, tool_input, decision.reason)
                if not approved:
                    results.append({
                        "type": "tool_result",
                        "tool_use_id": tool_call.id,
                        "content": "User denied permission",
                        "is_error": True
                    })
                    continue

            # Execute the tool
            try:
                result = await self._invoke_tool(tool_name, tool_input)
                results.append({
                    "type": "tool_result",
                    "tool_use_id": tool_call.id,
                    "content": json.dumps(result)
                })
            except Exception as e:
                self._log.error("tool_error", tool=tool_name, error=str(e))
                results.append({
                    "type": "tool_result",
                    "tool_use_id": tool_call.id,
                    "content": f"Error: {str(e)}",
                    "is_error": True
                })

        return results

    async def _invoke_tool(self, name: str, args: dict) -> Any:
        """Route tool call to appropriate MCP server."""
        # Robot tools
        if name in ROBOT_TOOLS:
            return await self._robot_mcp.call(name, args)

        # Memory tools
        if name in MEMORY_TOOLS:
            return await self._memory_mcp.call(name, args)

        raise ValueError(f"Unknown tool: {name}")

    # =========== Context Building ===========

    def _build_messages(self, user_input: str) -> list[dict]:
        """Build message list with context window."""
        messages = []

        # Add context window (recent turns)
        for turn in self._memory.get_context_window()[:-1]:  # Exclude current
            messages.append({
                "role": turn["role"],
                "content": turn["content"]
            })

        # Add current user input
        messages.append({
            "role": "user",
            "content": user_input
        })

        return messages

    def _load_system_prompt(self) -> str:
        """Load system prompt, optionally with persona overlay."""
        base_prompt = Path(self._config.system_prompt_path).read_text()

        if self._config.persona_path:
            persona_prompt = Path(self._config.persona_path).read_text()
            # Extract just the content after YAML frontmatter
            if "---" in persona_prompt:
                _, _, persona_content = persona_prompt.split("---", 2)
                base_prompt = f"{base_prompt}\n\n{persona_content.strip()}"

        return base_prompt

    # =========== Helpers ===========

    def _build_tool_list(self) -> list[dict]:
        """Build combined tool list from all MCP servers."""
        tools = []
        tools.extend(get_robot_tools())
        tools.extend(get_memory_tools())
        return tools

    def _extract_text(self, response: anthropic.types.Message) -> str:
        """Extract text content from response."""
        for block in response.content:
            if hasattr(block, "text"):
                return block.text
        return ""

    async def _confirm_tool(self, name: str, args: dict, reason: str | None) -> bool:
        """Ask user to confirm tool execution."""
        # In voice mode, use voice confirmation
        if self._voice and self._voice.is_active:
            await self._robot.speak(f"Can I {name}?")
            response = await self._robot.listen(timeout=10)
            return any(w in response.lower() for w in ["yes", "okay", "sure"])

        # In text mode, use terminal
        print(f"\n🔐 Permission Required: {name}")
        print(f"   Args: {json.dumps(args, indent=2)}")
        if reason:
            print(f"   Reason: {reason}")
        response = input("   Allow? [y/N]: ").strip().lower()
        return response in ("y", "yes")
```

## Interaction Modes

### Text Mode (REPL)

```python
async def run_text_mode(agent: ReachyAgentLoop):
    """Run agent in text REPL mode."""
    print("🤖 Reachy Agent (type 'quit' to exit)")

    while True:
        try:
            user_input = input("\n> ").strip()
            if user_input.lower() in ("quit", "exit"):
                break
            if not user_input:
                continue

            response = await agent.process(user_input)
            print(f"\n{response}")

        except KeyboardInterrupt:
            break

    print("\nGoodbye!")
```

### Voice Mode

```python
async def run_voice_mode(agent: ReachyAgentLoop):
    """Run agent in voice mode."""
    # Voice pipeline handles the loop
    # Agent.process() is called by voice event handlers

    print("🎤 Voice mode active (Ctrl+C to exit)")

    try:
        while agent._running:
            await asyncio.sleep(1)
    except KeyboardInterrupt:
        pass
```

### API Mode

```python
from fastapi import FastAPI

app = FastAPI()

@app.post("/chat")
async def chat(request: ChatRequest) -> ChatResponse:
    """HTTP endpoint for agent interaction."""
    response = await agent.process(request.message)
    return ChatResponse(response=response)
```

## Startup Sequence

```python
async def main():
    """Main entry point."""
    # Parse CLI args
    config = AgentConfig(
        enable_voice=args.voice,
        mock_hardware=args.mock
    )

    # Create and start agent
    agent = ReachyAgentLoop(config)

    try:
        await agent.start()

        # Run appropriate mode
        if config.enable_voice:
            await run_voice_mode(agent)
        else:
            await run_text_mode(agent)

    finally:
        await agent.stop()

if __name__ == "__main__":
    asyncio.run(main())
```

## Error Recovery

```python
class ReachyAgentLoop:
    async def process(self, user_input: str) -> str:
        """Process with error recovery."""
        try:
            return await self._process_internal(user_input)

        except anthropic.APIError as e:
            self._log.error("claude_api_error", error=str(e))
            return "I'm having trouble connecting to my brain. Let me try again."

        except ConnectionError as e:
            self._log.error("robot_connection_error", error=str(e))
            # Try to reconnect
            await self._reconnect_robot()
            return "I lost connection to my body briefly. I'm back now."

        except Exception as e:
            self._log.exception("unexpected_error")
            return "Something unexpected happened. Could you try again?"

    async def _reconnect_robot(self):
        """Attempt to reconnect to robot."""
        for attempt in range(3):
            try:
                await self._robot.connect()
                self._log.info("robot_reconnected", attempt=attempt)
                return
            except Exception:
                await asyncio.sleep(1)

        raise ConnectionError("Failed to reconnect to robot")
```

## Configuration

```yaml
# config/default.yaml
agent:
  model: claude-haiku-4-5-20251001
  name: Jarvis
  max_tokens: 4096
  temperature: 0.7

system_prompt: prompts/system.md

# Component flags controlled via CLI:
# --voice, --mock, etc.
```

## CLI Interface

```python
import typer

app = typer.Typer()

@app.command()
def run(
    voice: bool = typer.Option(False, "--voice", help="Enable voice mode"),
    mock: bool = typer.Option(False, "--mock", help="Use mock hardware"),
    debug_voice: bool = typer.Option(False, "--debug-voice", help="Debug voice events"),
    persona: str = typer.Option(None, "--persona", help="Persona to use")
):
    """Run the Reachy agent."""
    config = AgentConfig(
        enable_voice=voice,
        mock_hardware=mock,
        persona_path=f"prompts/personas/{persona}.md" if persona else None
    )
    asyncio.run(main(config))

@app.command()
def check():
    """Check system health."""
    # ... health check implementation

@app.command()
def version():
    """Show version."""
    print("reachy-agent v0.1.0")

if __name__ == "__main__":
    app()
```

## Related Specs

- [01-overview.md](./01-overview.md) - System architecture
- [02-robot-control.md](./02-robot-control.md) - Robot MCP tools
- [03-voice.md](./03-voice.md) - Voice pipeline (calls agent.process())
- [05-memory.md](./05-memory.md) - Memory and context window
- [06-permissions.md](./06-permissions.md) - Tool permission hooks

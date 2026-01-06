# Anthropic Python SDK Documentation

**Source:** https://github.com/anthropics/anthropic-sdk-python
**Scraped:** 2025-01-XX
**SDK Version:** Supports Python 3.9+

---

## Installation

```bash
pip install anthropic
```

For additional platform support:
```bash
# AWS Bedrock support
pip install anthropic[bedrock]

# Google Vertex support
pip install anthropic[vertex]
```

---

## Basic Usage

### Synchronous Client

```python
import os
from anthropic import Anthropic

client = Anthropic(
    api_key=os.environ.get("ANTHROPIC_API_KEY")
)

message = client.messages.create(
    max_tokens=1024,
    messages=[
        {
            "role": "user",
            "content": "Hello, Claude"
        }
    ],
    model="claude-sonnet-4-5-20250929"
)
print(message.content)
```

### Asynchronous Client

```python
import asyncio
from anthropic import AsyncAnthropic

client = AsyncAnthropic(
    api_key=os.environ.get("ANTHROPIC_API_KEY")
)

async def main():
    message = await client.messages.create(
        max_tokens=1024,
        messages=[
            {
                "role": "user",
                "content": "Hello, Claude"
            }
        ],
        model="claude-sonnet-4-5-20250929"
    )
    print(message.content)

asyncio.run(main())
```

---

## Key Features

### Streaming Responses

Stream responses for real-time interaction:

```python
from anthropic import Anthropic

client = Anthropic()

stream = client.messages.create(
    max_tokens=1024,
    messages=[{"role": "user", "content": "Hello, Claude"}],
    model="claude-sonnet-4-5-20250929",
    stream=True
)

for event in stream:
    print(event.type)
```

### Token Counting

Count tokens before making requests to manage costs:

```python
count = client.messages.count_tokens(
    model="claude-sonnet-4-5-20250929",
    messages=[{"role": "user", "content": "Hello, world"}]
)
print(count.input_tokens)  # 10
```

### Tool Use (Function Calling)

Enable Claude to use custom tools/functions:

```python
import json
from anthropic import Anthropic, beta_tool

client = Anthropic()

@beta_tool
def get_weather(location: str) -> str:
    """Get weather information for a location."""
    return json.dumps({
        "location": location,
        "temperature": "68°F",
        "condition": "Sunny"
    })

runner = client.beta.messages.tool_runner(
    max_tokens=1024,
    model="claude-sonnet-4-5-20250929",
    tools=[get_weather],
    messages=[{"role": "user", "content": "What is the weather in SF?"}]
)

for message in runner:
    print(message)
```

---

## Platform-Specific Integrations

### AWS Bedrock

```bash
pip install anthropic[bedrock]
```

```python
from anthropic import AnthropicBedrock

client = AnthropicBedrock()
message = client.messages.create(
    model="anthropic.claude-sonnet-4-5-20250929-v1:0",
    max_tokens=1024,
    messages=[{"role": "user", "content": "Hello!"}]
)
```

### Google Vertex AI

```bash
pip install anthropic[vertex]
```

```python
from anthropic import AnthropicVertex

client = AnthropicVertex()
message = client.messages.create(
    model="claude-sonnet-4@20250514",
    max_tokens=100,
    messages=[{"role": "user", "content": "Hello!"}]
)
```

---

## Best Practices

### API Key Management

Use environment variables for API keys:

```python
import os
from anthropic import Anthropic

# Set via environment variable
# export ANTHROPIC_API_KEY='your-api-key-here'

client = Anthropic(
    api_key=os.environ.get("ANTHROPIC_API_KEY")
)
```

Consider using `python-dotenv` for local development:

```bash
pip install python-dotenv
```

```python
from dotenv import load_dotenv
import os
from anthropic import Anthropic

load_dotenv()
client = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
```

### Type Safety

The SDK provides full type hints for IDE autocomplete and type checking:

```python
from anthropic import Anthropic
from anthropic.types import Message, MessageParam

client = Anthropic()

messages: list[MessageParam] = [
    {"role": "user", "content": "Hello!"}
]

response: Message = client.messages.create(
    model="claude-sonnet-4-5-20250929",
    max_tokens=1024,
    messages=messages
)
```

### Error Handling

Handle API errors appropriately:

```python
from anthropic import Anthropic, APIError, RateLimitError

client = Anthropic()

try:
    message = client.messages.create(
        model="claude-sonnet-4-5-20250929",
        max_tokens=1024,
        messages=[{"role": "user", "content": "Hello!"}]
    )
except RateLimitError as e:
    print(f"Rate limit exceeded: {e}")
except APIError as e:
    print(f"API error: {e}")
```

---

## Agent Patterns

### Conversational Agent

Maintain conversation history:

```python
from anthropic import Anthropic

client = Anthropic()
conversation_history = []

def chat(user_message: str) -> str:
    # Add user message to history
    conversation_history.append({
        "role": "user",
        "content": user_message
    })

    # Get response from Claude
    response = client.messages.create(
        model="claude-sonnet-4-5-20250929",
        max_tokens=1024,
        messages=conversation_history
    )

    # Add assistant response to history
    assistant_message = response.content[0].text
    conversation_history.append({
        "role": "assistant",
        "content": assistant_message
    })

    return assistant_message

# Usage
response1 = chat("What is Python?")
response2 = chat("Can you show me an example?")  # Has context from first message
```

### Async Agent with Streaming

Process responses as they arrive:

```python
import asyncio
from anthropic import AsyncAnthropic

async def streaming_agent(prompt: str):
    client = AsyncAnthropic()

    async with client.messages.stream(
        model="claude-sonnet-4-5-20250929",
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}]
    ) as stream:
        async for text in stream.text_stream:
            print(text, end="", flush=True)

    print()  # New line after stream completes

# Usage
asyncio.run(streaming_agent("Explain async programming"))
```

### Tool-Using Agent

Build an agent that can use multiple tools:

```python
from anthropic import Anthropic, beta_tool
import json

client = Anthropic()

@beta_tool
def calculator(operation: str, x: float, y: float) -> str:
    """Perform basic math operations."""
    ops = {
        "add": x + y,
        "subtract": x - y,
        "multiply": x * y,
        "divide": x / y if y != 0 else "Error: Division by zero"
    }
    return json.dumps({"result": ops.get(operation, "Invalid operation")})

@beta_tool
def get_current_time() -> str:
    """Get the current time."""
    from datetime import datetime
    return json.dumps({"time": datetime.now().isoformat()})

# Run agent with multiple tools
runner = client.beta.messages.tool_runner(
    max_tokens=1024,
    model="claude-sonnet-4-5-20250929",
    tools=[calculator, get_current_time],
    messages=[{
        "role": "user",
        "content": "What is 15 * 24 and what time is it?"
    }]
)

for message in runner:
    print(message)
```

---

## Performance Considerations

### Async for Concurrent Requests

Use async when making multiple requests:

```python
import asyncio
from anthropic import AsyncAnthropic

async def process_batch(prompts: list[str]):
    client = AsyncAnthropic()

    tasks = [
        client.messages.create(
            model="claude-sonnet-4-5-20250929",
            max_tokens=100,
            messages=[{"role": "user", "content": prompt}]
        )
        for prompt in prompts
    ]

    results = await asyncio.gather(*tasks)
    return results

# Process multiple prompts concurrently
prompts = ["Hello", "What is AI?", "Explain Python"]
results = asyncio.run(process_batch(prompts))
```

### Token Management

Monitor and optimize token usage:

```python
# Count tokens before sending
token_count = client.messages.count_tokens(
    model="claude-sonnet-4-5-20250929",
    messages=[{"role": "user", "content": "Your prompt here"}]
)

if token_count.input_tokens > 1000:
    print("Warning: Large prompt, consider truncating")

# Check response token usage
response = client.messages.create(
    model="claude-sonnet-4-5-20250929",
    max_tokens=1024,
    messages=[{"role": "user", "content": "Hello"}]
)

print(f"Input tokens: {response.usage.input_tokens}")
print(f"Output tokens: {response.usage.output_tokens}")
```

---

## Requirements

- Python 3.9 or higher
- `anthropic` package from PyPI
- Valid Anthropic API key

## Additional Resources

- GitHub Repository: https://github.com/anthropics/anthropic-sdk-python
- API Documentation: https://platform.claude.com/docs
- Community Support: Anthropic Discord and forums

---

## Summary

The Anthropic Python SDK provides:
- ✅ Type-safe request/response handling
- ✅ Synchronous and asynchronous clients
- ✅ Streaming support for real-time responses
- ✅ Tool use (function calling) capabilities
- ✅ Token counting for cost management
- ✅ Multi-platform support (Bedrock, Vertex)
- ✅ Comprehensive error handling
- ✅ Python 3.9+ compatibility

Use this SDK for building conversational AI agents, tool-using applications, and integrated AI workflows with Claude models.

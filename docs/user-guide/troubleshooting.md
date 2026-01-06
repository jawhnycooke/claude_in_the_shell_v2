# Troubleshooting Guide

Solutions to common issues with Claude in the Shell v2.

## Quick Diagnostics

Run the health check first:

```bash
python -m reachy_agent check
```

This verifies:
- Python version
- API keys
- ChromaDB initialization
- Configuration loading
- Robot/simulation connection

## Installation Issues

### "ModuleNotFoundError: No module named 'reachy_agent'"

**Cause**: Package not installed in editable mode

**Solution**:
```bash
# Reinstall with -e flag
uv pip install -e .
# Or
pip install -e .
```

### "uv: command not found"

**Cause**: uv not installed

**Solution**:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
# Restart terminal or source profile
```

### ChromaDB SQLite Version Error

**Cause**: System SQLite too old

**Solution**:
```bash
pip install pysqlite3-binary
```

Then add to your script before imports:
```python
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
```

### PyAudio Build Failure

**Linux**:
```bash
sudo apt-get install portaudio19-dev python3-pyaudio
pip install pyaudio
```

**macOS**:
```bash
brew install portaudio
pip install pyaudio
```

**Windows**:
```bash
pip install pipwin
pipwin install pyaudio
```

## API Key Issues

### "ANTHROPIC_API_KEY not found"

**Check if set**:
```bash
echo $ANTHROPIC_API_KEY
```

**Set in environment**:
```bash
export ANTHROPIC_API_KEY=sk-ant-...
```

**Or add to .env file**:
```bash
echo "ANTHROPIC_API_KEY=sk-ant-..." >> .env
```

### "OpenAI API key not found" (Voice Mode)

**Solution**:
```bash
# Add to .env
echo "OPENAI_API_KEY=sk-..." >> .env
```

### "Invalid API key"

1. Check key is correct (no extra spaces)
2. Verify key hasn't expired
3. Check API dashboard for issues

## Connection Issues

### "Robot connection failed"

**For real hardware**:
1. Check robot is powered on
2. Verify WiFi connection
3. Check IP address in config
4. Try pinging the robot

**Quick fix**: Use mock mode
```bash
python -m reachy_agent run --mock
```

### "SDK initialization failed"

1. Check Reachy Mini SDK is installed
2. Verify Zenoh service is running
3. Check firewall settings

### "WebSocket connection failed" (Voice)

1. Check internet connection
2. Verify OpenAI API key
3. Check firewall allows WebSocket

## Voice Issues

### Wake Word Not Detected

**Symptoms**: Say wake word, nothing happens

**Diagnostics**:
```bash
python -m reachy_agent run --mock --voice --debug-voice
```

Look for:
```
voice_event event=wake_detected confidence=0.XX
```

**Solutions**:
1. **Low confidence**: Increase `wake_sensitivity` in config
2. **No detection**: Check microphone is working
3. **Wrong device**: Check default audio input

**Test microphone**:
```bash
# Linux
arecord -d 3 test.wav && aplay test.wav

# macOS
# Use QuickTime or System Preferences > Sound
```

### Poor Transcription Quality

**Symptoms**: Text doesn't match speech

**Solutions**:
1. Speak more clearly
2. Reduce background noise
3. Get closer to microphone
4. Check internet connection (STT is cloud-based)

### Audio Output Issues

**No sound from robot**:
1. Check speaker/volume settings
2. Verify audio device selection
3. Test with system sounds

**Distorted audio**:
1. Lower volume
2. Check sample rate compatibility

### Barge-in Not Working

**Symptoms**: Can't interrupt robot while speaking

**Solutions**:
1. Say complete wake word ("Hey Jarvis" not just "Hey")
2. Speak at normal volume
3. Wait for speech to start before interrupting

## Memory Issues

### "ChromaDB initialization failed"

**Reset memory database**:
```bash
rm -rf ~/.reachy/memory
# Restart agent
```

### Search Returns No Results

1. Check search query is meaningful
2. Try removing type filter
3. Lower `min_score` threshold
4. Verify memories were actually stored

### Duplicate Memories

The system doesn't auto-deduplicate. Ask the agent:
```
> Check if you already know this before storing
```

### Memory Not Persisting

1. Check write permissions to `~/.reachy/memory`
2. Verify ChromaDB initialized correctly
3. Check for errors in logs

## Motion Issues

### Robot Not Moving

**Check wake state**:
```
> Are you awake?
> Wake up
```

**Check in mock mode**:
Movements are simulated, check logs for confirmation.

### Jerky Movement

1. Increase movement duration
2. Check motion tick rate (should be 30Hz)
3. Reduce idle behavior amplitude

### Movements Hitting Limits

Joint limits are automatically clamped. Check:
```
> What are your movement limits?
```

## Simulation Issues

### "MuJoCo not found"

```bash
pip install gymnasium[mujoco]
# Or
pip install mujoco
```

### Viewer Window Not Opening

**Check display**:
```bash
echo $DISPLAY
```

**For SSH sessions**:
```bash
ssh -X user@host
```

**Use headless mode**:
```bash
export MUJOCO_GL=egl
```

### Simulation Running Slowly

1. Disable viewer: Remove `--sim-viewer`
2. Use fast-forward: `--no-sim-realtime`
3. Reduce render quality in config

### Physics Instability

1. Increase physics substeps in config
2. Reduce PD controller gains
3. Check for joint limit violations

## Permission Issues

### "Permission denied" for tools

Tool requires CONFIRM permission. You'll see:
```
🔐 Permission Required
   Tool: store_memory
   Allow? [y/N]:
```

Type `y` to allow.

### Tool Forbidden

Some tools are FORBIDDEN by design (e.g., shell commands).

Check `config/permissions.yaml` for rules.

## Performance Issues

### High Memory Usage

1. Reduce context window size
2. Clean up old memories
3. Restart agent periodically

### Slow Response Time

1. Check internet connection (API calls)
2. Use Claude Haiku (faster than Sonnet)
3. Reduce max_tokens if responses are too long

### High CPU Usage

1. Disable voice when not needed
2. Reduce motion tick rate
3. Use simulation only when needed

## Logging and Debugging

### Enable Debug Logging

```bash
# Voice debugging
python -m reachy_agent run --mock --voice --debug-voice

# General debug
export REACHY_LOG_LEVEL=DEBUG
python -m reachy_agent run --mock
```

### Check Logs

Logs are output to stderr. Redirect to file:
```bash
python -m reachy_agent run --mock 2> agent.log
```

### Audit Log

Permission decisions are logged to:
```
~/.reachy/audit.jsonl
```

View recent entries:
```bash
tail -20 ~/.reachy/audit.jsonl | jq .
```

## Getting Help

### Collect Diagnostic Info

```bash
# System info
python --version
pip list | grep -E "anthropic|chromadb|mujoco"

# Run health check
python -m reachy_agent check

# Test basic functionality
python -c "from reachy_agent import __version__; print(__version__)"
```

### Report Issues

When reporting bugs, include:
1. Python version
2. Operating system
3. Output of `python -m reachy_agent check`
4. Error message and traceback
5. Steps to reproduce

### Community Support

- **GitHub Issues**: Bug reports and feature requests
- **Discussions**: Questions and ideas
- **Documentation**: You're reading it!

---

**Still stuck?** Check the [API Reference](../api-reference/index.md) or [Developer Guide](../developer-guide/index.md) for more details.

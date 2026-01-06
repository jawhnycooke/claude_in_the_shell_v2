# Tutorial: Voice Interaction

Set up voice control with wake words and have natural conversations.

**Time**: 15 minutes
**Prerequisites**: Voice dependencies installed, OpenAI API key configured

## What You'll Learn

- Set up voice mode
- Use wake words to activate different personas
- Have natural voice conversations
- Interrupt with barge-in
- Debug voice issues

## Step 1: Install Voice Dependencies

If you haven't already:

```bash
uv pip install -e ".[voice]"
```

Verify the installation:
```bash
python -c "import pyaudio; import openwakeword; print('Voice deps OK')"
```

## Step 2: Configure OpenAI API Key

Voice mode requires OpenAI for speech-to-text and text-to-speech:

```bash
# Add to your .env file
echo "OPENAI_API_KEY=sk-..." >> .env
```

Verify:
```bash
python -m reachy_agent check
```

Should show:
```
✅ OpenAI API: Connected
```

## Step 3: Start Voice Mode

Launch the agent with voice enabled:

```bash
python -m reachy_agent run --mock --voice
```

You should see:
```
🎤 Voice mode active (Ctrl+C to exit)
🔊 Listening for wake words: hey_jarvis, hey_motoko, hey_batou
```

## Step 4: Use Your First Wake Word

Say clearly: **"Hey Jarvis"**

You should hear/see:
1. A confirmation sound or visual indicator
2. The robot starts listening
3. Say your command: "Hello, who are you?"
4. Wait for the response

```
[Voice] Wake detected: jarvis (confidence: 0.87)
[Voice] Listening...
[Voice] Transcribed: "Hello, who are you?"
[Voice] Speaking response...

"Hello! I'm Jarvis, an AI assistant inhabiting a Reachy Mini robot..."
```

## Step 5: Try Different Personas

Each wake word activates a different personality:

### Jarvis (Default)
- **Wake word**: "Hey Jarvis"
- **Voice**: Echo (male, warm)
- **Personality**: Professional, helpful assistant

Say: "Hey Jarvis, tell me a fun fact"

### Motoko
- **Wake word**: "Hey Motoko"
- **Voice**: Nova (female, professional)
- **Personality**: Analytical, philosophical

Say: "Hey Motoko, what do you think about consciousness?"

### Batou
- **Wake word**: "Hey Batou"
- **Voice**: Onyx (male, deep)
- **Personality**: Direct, action-oriented

Say: "Hey Batou, give me a workout tip"

## Step 6: Have a Conversation

Voice mode maintains context across turns:

```
You: "Hey Jarvis"
Jarvis: *listening*

You: "My name is Alex"
Jarvis: "Nice to meet you, Alex!"

You: "Hey Jarvis"
Jarvis: *listening*

You: "What's my name?"
Jarvis: "Your name is Alex, as you just told me."
```

## Step 7: Try Barge-in (Interruption)

While the robot is speaking, say the wake word to interrupt:

```
You: "Hey Jarvis"
Jarvis: *listening*

You: "Tell me a long story about robots"
Jarvis: "Once upon a time, in a factory far away, there lived a small robot named Bolt. Bolt was different from the other robots because he had a curious nature that led him to explore the factory at night. One evening, as he was wandering through the assembly line, he discovered a hidden room filled with..."

You: "Hey Jarvis"  (interrupting)
Jarvis: *stops speaking immediately*
Jarvis: *listening*

You: "Actually, just tell me a short joke instead"
Jarvis: "Why did the robot go on vacation? Because it needed to recharge its batteries!"
```

## Step 8: Control the Robot with Voice

Use voice commands for movement:

```
You: "Hey Jarvis"
You: "Look up at the ceiling"
*robot moves head up*

You: "Hey Jarvis"
You: "Turn around slowly"
*robot body rotates 180°*

You: "Hey Jarvis"
You: "Show me you're happy"
*robot plays happy emotion*
```

## Step 9: Debug Voice Issues

If wake words aren't being detected, enable debug mode:

```bash
python -m reachy_agent run --mock --voice --debug-voice
```

This shows detailed event logging:

```
[12:34:56.123] voice_event event=wake_detected persona=jarvis confidence=0.87
[12:34:56.125] voice_event event=listening_start persona=jarvis
[12:34:58.456] voice_event event=listening_end audio_duration=2.33
[12:34:58.789] voice_event event=transcribed text="Hello" confidence=0.94
[12:34:58.791] voice_event event=processing text="Hello"
[12:34:59.234] voice_event event=response text="Hello! How can I..."
[12:34:59.236] voice_event event=speaking_start voice=echo
[12:35:01.567] voice_event event=speaking_end
```

## Common Issues and Solutions

### Wake Word Not Detecting

**Symptoms**: Say wake word, nothing happens

**Try**:
1. Speak clearer and louder
2. Get closer to the microphone
3. Reduce background noise
4. Increase sensitivity in config:
   ```yaml
   voice:
     wake_sensitivity: 0.6  # Default is 0.5
   ```

### Poor Transcription

**Symptoms**: Robot misunderstands what you say

**Try**:
1. Speak slower and clearer
2. Check internet connection (STT is cloud-based)
3. Reduce background noise

### No Audio Output

**Symptoms**: Robot doesn't speak

**Try**:
1. Check speaker volume
2. Verify audio output device
3. Test with system sounds

## Configuration Options

Customize voice behavior in `config/default.yaml`:

```yaml
voice:
  personas: [motoko, batou, jarvis]  # Available personas
  wake_sensitivity: 0.5              # 0.0-1.0
  silence_threshold: 0.3             # Seconds to wait for speech end
  max_listen_time: 30.0              # Max listening duration

openai:
  realtime_model: gpt-4o-realtime-preview
```

## Creating Custom Personas

Add new personas by creating files in `prompts/personas/`:

```markdown
# prompts/personas/alfred.md
---
name: Alfred
wake_word: hey_alfred
voice: fable
---

# Alfred the Butler

You are Alfred, a distinguished AI butler with impeccable manners.
You speak formally and always address the user as "Sir" or "Madam".
You have a dry wit and occasionally make subtle jokes.
```

The persona will be auto-discovered on restart.

## Tips for Best Results

1. **Quiet environment**: Background noise affects detection
2. **Consistent distance**: Stay 30-60cm from microphone
3. **Clear pronunciation**: Enunciate wake words clearly
4. **Pause after wake word**: Wait for listening indicator
5. **Natural speed**: Don't speak too fast or slow

## What's Next?

You've mastered voice interaction! Continue to:

- [Emotion Expressions](emotion-expressions.md) - Express emotions
- [Voice Control Guide](../user-guide/voice-control.md) - Complete reference
- [Custom Behaviors](custom-behaviors.md) - Create new behaviors

## Summary

In this tutorial, you learned to:

- [x] Install voice dependencies
- [x] Configure OpenAI API
- [x] Start voice mode
- [x] Use wake words (Jarvis, Motoko, Batou)
- [x] Have multi-turn conversations
- [x] Interrupt with barge-in
- [x] Control the robot with voice
- [x] Debug voice issues
- [x] Create custom personas

---

**Congratulations!** You've completed the voice interaction tutorial.

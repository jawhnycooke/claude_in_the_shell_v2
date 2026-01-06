# Wake Word Models

This directory contains custom wake word models for OpenWakeWord.

## Default Models

The system uses OpenWakeWord's built-in models by default:
- `hey_jarvis`
- `hey_motoko` (custom, needs training)
- `hey_batou` (custom, needs training)

## Custom Training

To train custom wake words:

1. Collect audio samples (50+ positive, 100+ negative)
2. Use OpenWakeWord training tools
3. Place resulting `.tflite` or `.onnx` model here
4. Update persona files to reference model

See OpenWakeWord documentation for details:
https://github.com/dscripka/openWakeWord

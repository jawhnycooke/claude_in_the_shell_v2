# Emotion Library

This directory contains 81 emotion animations for the Reachy Mini robot.

## Downloading Emotions

Run the download script:

```bash
python scripts/download_emotions.py
```

This will fetch all emotion animations from the Pollen Robotics repository.

## Format

Emotions are stored as `.npz` files with the following structure:

- `frames`: Array of joint positions (pitch, yaw, roll, z, antenna_left, antenna_right)
- `fps`: Frames per second (typically 30)
- `duration`: Animation duration in seconds

## Categories

- **Basic**: happy, sad, curious, surprised, confused, tired
- **Social**: greeting, farewell, acknowledgment, thinking
- **Reactions**: yes, no, maybe, excitement, disappointment
- **Complex**: interested, skeptical, amused, concerned

See `manifest.json` for the complete list with metadata.

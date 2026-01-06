# MCP Tools Quick Reference

## Robot Control (20 tools)

### Movement
- `move_head(pitch, yaw, roll, z, duration)` - Move to absolute position
- `look_at(x, y, z, duration)` - Look at 3D point
- `rotate_body(angle, duration)` - Rotate body
- `reset_position(duration)` - Return to neutral

### Expression
- `play_emotion(name)` - Play emotion animation
- `play_sequence(emotions[], delays[])` - Play emotion sequence
- `set_antennas(left, right)` - Set antenna positions
- `nod(intensity)` - Nod yes
- `shake(intensity)` - Shake no

### Audio
- `speak(text, voice)` - Text-to-speech
- `listen(timeout)` - Speech-to-text

### Perception
- `capture_image(format)` - Get camera frame
- `get_sensor_data()` - Read IMU
- `detect_sound_direction()` - Sound localization

### Lifecycle
- `wake_up()` - Enable motors
- `sleep()` - Disable motors
- `is_awake()` - Check motor state

### Status (cached 200ms)
- `get_status()` - Complete robot state
- `get_pose()` - Current head position
- `get_battery()` - Battery level

## Memory (3 tools)

- `search_memories(query, type?, limit)` - Semantic search
- `store_memory(content, type, metadata?)` - Store new memory
- `forget_memory(id)` - Delete memory

### Memory Types
- `conversation` - 30 day expiry
- `fact` - Permanent
- `context` - Same day expiry

## Permissions

- `get_*`, `is_*`, `detect_*` - AUTONOMOUS
- `move_*`, `play_*`, `speak`, `listen` - AUTONOMOUS
- `store_memory`, `forget_memory` - CONFIRM
- `exec_*`, `shell_*` - FORBIDDEN

See `config/permissions.yaml` for full rules.

# Tutorial: First Robot Movement

Learn to control the Reachy Mini robot's head, body, and antennas.

**Time**: 10 minutes
**Prerequisites**: Installation complete, API key configured

## What You'll Learn

- Move the robot's head in different directions
- Rotate the body 360°
- Control the antennas for expression
- Use natural language commands
- Check robot status and position

## Step 1: Start the Agent

Open your terminal and start the agent in mock mode:

```bash
python -m reachy_agent run --mock
```

You should see:
```
🤖 Reachy Agent (type 'quit' to exit)

>
```

:material-lightbulb: **Tip**: Mock mode simulates the robot without hardware. All movements are logged but no physical robot moves.

## Step 2: Wake Up the Robot

Before moving, the robot must be "awake" (motors enabled):

```
> Wake up

I'm now awake and ready! My motors are enabled.
```

Check the status:
```
> What's your status?

Here's my current status:
- Awake: Yes
- Battery: 100%
- Head position: pitch=0°, yaw=0°, roll=0°, z=0mm
- Body angle: 0°
- Antennas: left=0°, right=0°
```

## Step 3: Basic Head Movements

### Look Up and Down (Pitch)

```
> Look up

*moving head*
I've tilted my head up (pitch: 20°).

> Look down

*moving head*
I've tilted my head down (pitch: -25°).
```

### Look Left and Right (Yaw)

```
> Look to the left

*moving head*
I've turned my head to the left (yaw: -40°).

> Look right

*moving head*
I've turned my head to the right (yaw: 40°).
```

### Tilt Head (Roll)

```
> Tilt your head curiously

*moving head*
I've tilted my head (roll: 15°). Like this?
```

### Combined Movement

```
> Look up and to the left

*moving head*
I've moved my head to look up-left (pitch: 15°, yaw: -30°).
```

## Step 4: Look At a Point

Instead of specifying angles, you can tell the robot to look at a point in space:

```
> Look at something about 1 meter ahead of you

*calculating and moving*
I'm now looking at a point 1 meter ahead.

> Look at the ground in front of you

*moving head*
I'm looking at the ground ahead (approximately 0.5m forward, 0.3m down).
```

## Step 5: Body Rotation

The body can rotate 360° continuously:

```
> Turn around to face the other way

*rotating body*
I've rotated 180° to face the opposite direction.

> Turn back to your original position

*rotating body*
Back to facing forward (0°).

> Rotate 90 degrees to the right

*rotating body*
I've rotated 90° clockwise.
```

## Step 6: Antenna Control

The antennas are great for expression:

```
> Raise your antennas excitedly

*moving antennas*
Both antennas are up! (left: +60°, right: +60°)

> Droop them sadly

*moving antennas*
*sigh* (left: -90°, right: -90°)

> Make them asymmetric - left up, right down

*moving antennas*
Quirky look activated! (left: +45°, right: -45°)
```

## Step 7: Gestures

### Nodding (Yes)

```
> Nod yes

*nodding*
*nods affirmatively*

> Nod enthusiastically

*nodding more*
*enthusiastic nodding*
```

### Shaking (No)

```
> Shake your head no

*shaking*
*shakes head disapprovingly*

> Shake emphatically

*shaking more*
*emphatic head shake*
```

## Step 8: Reset Position

Return to neutral pose:

```
> Reset to neutral position

*resetting*
I've returned to my neutral pose - head centered, body facing forward, antennas neutral.
```

## Step 9: Check Your Position

At any time, ask for the current position:

```
> What's your current position?

My current pose:
- Head: pitch=0°, yaw=0°, roll=0°, z=0mm
- Body rotation: 0°
- Antennas: left=0°, right=0°
```

## Challenge: Create a Sequence

Try combining movements naturally:

```
> Look left curiously, then look right, then look up and nod

*executing sequence*
1. Looking left curiously (yaw: -30°, roll: 8°)
2. Looking right (yaw: 40°)
3. Looking up and nodding (pitch: 15°, then nod)

Done! That was fun.
```

## Movement Limits Reference

| Joint | Range | Notes |
|-------|-------|-------|
| Head Pitch | -45° to +35° | Down to up |
| Head Yaw | -60° to +60° | Right to left |
| Head Roll | -35° to +35° | Tilt |
| Head Z | 0 to 50mm | Height |
| Body | 0° to 360° | Continuous |
| Antennas | -150° to +150° | Each |

## What's Next?

You've learned the basics of robot movement! Continue to:

- [Emotion Expressions](emotion-expressions.md) - Make the robot more expressive
- [Voice Interaction](voice-interaction.md) - Control with your voice
- [Robot Movements Guide](../user-guide/robot-movements.md) - Complete reference

## Summary

In this tutorial, you learned to:

- [x] Wake up the robot
- [x] Move the head (pitch, yaw, roll)
- [x] Look at points in space
- [x] Rotate the body
- [x] Control antennas
- [x] Use gestures (nod, shake)
- [x] Reset to neutral
- [x] Check position and status

---

**Congratulations!** You've completed your first robot movement tutorial.

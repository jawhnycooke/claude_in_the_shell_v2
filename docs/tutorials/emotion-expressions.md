# Tutorial: Emotion Expressions

Make your robot expressive with 81 emotion animations.

**Time**: 10 minutes
**Prerequisites**: Basic movement tutorial completed

## What You'll Learn

- Play individual emotions
- Create emotion sequences
- Combine emotions with movement
- Use emotions contextually

## Step 1: Start the Agent

```bash
python -m reachy_agent run --mock
```

## Step 2: Basic Emotions

The robot has 81 bundled emotions. Start with the basics:

### Happy
```
> Show me you're happy

*plays happy animation*
😊 I'm feeling great!
```

### Sad
```
> Look sad

*plays sad animation*
😢 *sighs*
```

### Curious
```
> Be curious about something

*plays curious animation - head tilts, antennas perk up*
Hmm, what's that?
```

### Surprised
```
> Act surprised

*plays surprised animation - quick head back, antennas up*
Oh! I wasn't expecting that!
```

### Thinking
```
> Show me you're thinking

*plays thinking animation - slow head movement, antennas drift*
Let me ponder that...
```

### Confused
```
> Look confused

*plays confused animation - head tilt, asymmetric antennas*
I'm not sure I understand...
```

## Step 3: Emotion Categories

### Basic Emotions
| Emotion | Description | Use When |
|---------|-------------|----------|
| happy | Joyful expression | Good news, success |
| sad | Downcast expression | Bad news, empathy |
| curious | Interested, alert | Questions, exploration |
| surprised | Startled reaction | Unexpected information |
| confused | Puzzled look | Unclear situation |
| tired | Weary expression | Long conversation |

### Social Emotions
```
> Greet me

*plays greeting animation*
Hello there! Great to see you!

> Say goodbye

*plays farewell animation*
Until next time!

> Acknowledge what I said

*plays acknowledgment animation*
I understand.
```

### Reaction Emotions
```
> Show excitement

*plays excitement animation*
This is amazing!

> Show disappointment

*plays disappointment animation*
Oh, that's too bad...

> Express maybe/uncertainty

*plays maybe animation*
I'm not entirely sure about that...
```

### Complex Emotions
```
> Look interested

*plays interested animation*
Tell me more!

> Be skeptical

*plays skeptical animation*
Hmm, I'm not convinced...

> Look amused

*plays amused animation*
Ha! That's funny.

> Show concern

*plays concerned animation*
Are you okay?
```

## Step 4: Emotion Sequences

Chain emotions for richer expression:

### Simple Sequence
```
> First look curious, then surprised, then happy

*plays sequence*
1. *curious* - "Hmm, what's this..."
2. *surprised* - "Oh!"
3. *happy* - "I love it!"
```

### With Delays
```
> Look thoughtful for 2 seconds, then show understanding

*plays thinking for longer*
*then plays acknowledgment*
Ah, I see now!
```

### Narrative Sequence
```
> Show me the emotions of discovering a gift: curious, surprised, then excited

*plays discovery sequence*
1. *curious* - notices something
2. *surprised* - unwraps it
3. *excited* - loves the gift!
```

## Step 5: Emotions + Movement

Combine emotions with physical movement:

```
> Look curious and tilt your head to the side

*plays curious with head roll*
What's over there?

> Nod happily

*plays happy + nod*
Yes, I agree completely!

> Shake your head and look disappointed

*plays disappointed + shake*
No, that won't work...

> Look up excitedly with raised antennas

*looks up + excited + antennas up*
Look at that!
```

## Step 6: Contextual Emotions

The agent automatically chooses appropriate emotions:

```
> Tell me a joke

*plays thinking*
Hmm, let me think of a good one...
*plays amused*
Why don't scientists trust atoms? Because they make up everything!
*plays happy*

> I'm feeling down today

*plays concerned*
I'm sorry to hear that.
*plays empathetic*
What's troubling you?

> I got a promotion!

*plays excited*
Congratulations! That's wonderful news!
*plays happy*
You must be so proud!
```

## Step 7: Building Expressiveness

### Match Emotion to Content
```
> Say "I'm so happy for you" with an appropriate emotion

*plays happy + excited antenna movement*
"I'm so happy for you!"
```

### Transitions
```
> Go from confused to understanding to happy

*smooth transition through emotions*
1. *confused* - "I don't get it..."
2. *thinking* - "Wait..."
3. *understanding* - "Oh!"
4. *happy* - "I see now!"
```

### Intensity
```
> Show mild curiosity

*subtle curious animation*

> Show intense curiosity

*exaggerated curious animation with bigger movements*
```

## Available Emotions Reference

### Full List by Category

**Basic (6)**
- happy, sad, curious, surprised, confused, tired

**Social (4)**
- greeting, farewell, acknowledgment, thinking

**Reactions (5)**
- yes, no, maybe, excitement, disappointment

**Complex (4)**
- interested, skeptical, amused, concerned

Plus 62 more variations and combinations in `data/emotions/manifest.json`.

## Challenge: Create a Conversation Scene

Act out this scenario with appropriate emotions:

```
> Let's role play. You're meeting someone for the first time.

*greeting emotion*
"Hello! Nice to meet you!"

> They tell you an interesting fact

*curious → interested*
"Really? Tell me more!"

> They say something you disagree with

*thinking → skeptical*
"Hmm, I'm not sure about that..."

> They explain and you understand

*listening → acknowledgment → happy*
"Ah, I see what you mean now!"
```

## Tips for Natural Expression

1. **Don't overdo it**: Not every sentence needs an emotion
2. **Match intensity**: Subtle emotions for small things
3. **Use transitions**: Smooth changes between emotions
4. **Context matters**: Same words, different emotions = different meaning
5. **Combine wisely**: Movement + emotion + speech = full expression

## What's Next?

You've learned emotional expression! Continue to:

- [MuJoCo Setup](mujoco-setup.md) - Simulate with physics
- [Custom Behaviors](custom-behaviors.md) - Create your own emotions
- [Motion Guide](../user-guide/robot-movements.md) - Complete movement reference

## Summary

In this tutorial, you learned to:

- [x] Play basic emotions (happy, sad, curious, etc.)
- [x] Use emotion categories (basic, social, reactions, complex)
- [x] Create emotion sequences
- [x] Combine emotions with movement
- [x] Use contextual emotions
- [x] Build expressive interactions

---

**Congratulations!** You've completed the emotion expressions tutorial.

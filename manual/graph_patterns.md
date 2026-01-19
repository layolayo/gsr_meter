# Graph Patterns (Technical Reference)

The system transforms the classic "Needle" movements into a real-time **Digital Graph Trace**. 

Unlike the analog E-Meter where a read was fleeting ("if you didn't see it, it was gone"), our digital system provides a **Historical View**. You can see the shape, trajectory, and context of every reaction over time.

## Digital Advantages

1.  **History**: The scrolling graph allows you to catch reads you might have missed in the moment.
2.  **Context**: You can see if a "Fall" was part of a longer trend or an isolated reaction.
3.  **Precision**: The system mathematically calculates velocity, removing subjectivity.

## Understanding Tone Arm (TA)
The "Tone Arm" (TA) represents the **Basal Skin Resistance** (BSR) of the subject.

> **Basal Skin Resistance (BSR)**: The "Baseline" or slow-moving electrical resistance of the skin. It reflects the general state of arousal or tension (Tonic Level), as opposed to the rapid changes (Phasic) caused by immediate thoughts.

### 1. Mathematical Meaning (Ohms)
*   **Definition**: The raw electrical resistance of the body.
*   **Scale**: The range is typically **5,000 Ohms (TA 2.0)** to **12,500 Ohms (TA 3.0)**, but can extend to **78,125 Ohms (TA 5.0)**. The maximum is **TA 6.5**.
*   **Relationship**:
    *   **High TA** = High Resistance (**Dry Skin** / **Tight Graph**).
    *   **Low TA** = Low Resistance (**Moist Skin** / **Loose Graph**).

### 2. Physiological Meaning (Tension)
*   **Definition**: The physical tension or relaxation of the body.
*   **High TA**: Corresponds to physical tension, anxiety, or "charge".
*   **Low TA**: Corresponds to exhaustion or deep relaxation.

### 3. Psychological Meaning (Mass)
*   **Definition**: The "Mental Mass" or solidity in the mind.
*   **High TA**: The mind is solid, heavy, or "stuck" on a problem.
*   **Low TA**: The mind is clear, thin, or empty.
*   **Motion**: We want the TA to be **In Motion**. A stuck TA means a stuck mind.

## Units & Scaling
The system uses **Graph Units** (Relative Scale) to define patterns, ensuring consistency regardless of your current Sensitivity.

*   **Graph Range**: The active vertical display represents **5.0 Units**.
*   **Sensitivity**: As you adjust sensitivity (Window Size), the value of "1 Unit" changes in raw Ohms, but the *visual size* of a reaction stays consistent on the screen.

## Velocity & Magnitude
The system calculates the **Trace Velocity** in **Units per Second**.

*   **Steady Trend**: Slow movement (< 0.15 Units/sec).
*   **Instant Read**: Fast, sudden movement (> 0.15 Units/sec).

## Classified Patterns

The system automatically detects and labels these patterns based on their size in **Graph Units**:

| Pattern | Definition | Meaning |
| :--- | :--- | :--- |
| **FALL** | Drop > 1/8th Unit/sec (Steady) | Growing relaxation, Release of charge. |
| **RISE** | Rise > 1/8th Unit/sec (Steady) | Growing tension, Resistance. |
| **TICK** | Rapid Drop (1/8th - 1/5th Unit) | Small, distinct instant reaction. |
| **SHORT FALL** | Rapid Drop (1/5th - 1.5 Units) | Instant reaction to stimulus. |
| **LONG FALL** | Rapid Drop (> 1.5 Units) | Major release of charge (approx 1/3 screen). |
| **SHORT RISE** | Rapid Rise (1/5th -> 1.5 Units) | Sudden inhibition (mental brake) or resistance. |
| **LONG RISE** | Rapid Rise (> 1.5 Units) | Major tension or resistance (approx 1/3 screen). |
| **BLOWDOWN** | Extremely fast drop (> 2.0 Units/sec) | "Total" release. |
| **ROCKET READ** | Extremely fast rise (> 2.0 Units/sec) | Immediate reactive defense. |
| **FLOATING WAVE** | Rhythmic sine-wave motion | **Positive Indicator** - Good place / Release. |

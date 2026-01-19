# Device Calibration Guide

This document explains the technical and historical context of the device calibration process used in the GSR Meter application.

## The Concept

The core measurement of the device is **Tone Arm (TA)**, which corresponds to the static resistance (Basal Skin Response) of the user. Changes in this resistance are the primary signal for biofeedback.

### Sensitivity
The "Sensitivity" of the device determines how much the graph moves in response to a change in the user's state.

In our digital implementation, **Sensitivity** is dynamically calculated. We use a **Calibration Wizard** to determine the optimal sensitivity window based on the user's physiological range.

## The Calibration Wizard

The app uses a 4-step calibration process to find your **Responsiveness Threshold**.

### Step 1-3: The Squeeze Test
The first three steps measure your **Active Response Range** using a "Squeeze" mechanism. This mimics the "Can Squeeze" drill to ensure the device is reading correctly.

1.  **Instruction**: "SQUEEZE SENSOR"
2.  **Detection**: The system waits for a **Drop** in TA (resistance decreases as sweat glands activate).
3.  **Release**: The system waits for the TA to recover.

This is repeated 3 times to get a reliable median value of your drop magnitude. This median is then used to set the **Base Sensitivity** of the graph.

### Step 4: The Deep Breath Test
The final step measures your **Autonomic Response** (Rest & Digest).

1.  **Instruction**: "DEEP BREATH IN"
2.  **Detection**: The system waits for a drop triggered by the diaphragm movement and relaxation.
3.  **Stabilization**: Instead of a quick release, the system waits for the reading to **Stabilize** (flatline) for 1.5 seconds.

This ensures that the "Sensitivity Knob" of our virtual GSR-Meter is set perfectly for your current session, neither too loose (random noise) nor too stiff (missing reads).

## Graph Patterns

For a detailed technical reference on graph patterns (Falls, Rises, Rocket Reads, etc.), please see the **[Graph Patterns Guide](manual/graph_patterns.md)**.

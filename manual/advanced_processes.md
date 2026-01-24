# Using Automated Sessions

The **Processes** feature allows you to run pre-scripted, automated sessions. The system guides you through the session step-by-step, playing audio for each question and managing the calibration automatically.

## Starting an Automated Session

1.  **Launch the System**: Start the main application (`v44_presession.py`).
2.  **Verify Mic**: Ensure your microphone is active (check the "Mic Verified" status in the log).
3.  **Click "Processes"**: Locate the "Processes" button in the control panel.
4.  **Select a Process**: A dialog will appear listing available processes (e.g., "Issue Buster", "ARC Straightwire").
    *   *Note*: Some processes may ask for a specific item to process (e.g., selecting a specific definition or subject).
5.  **Enter Notes**: You can enter pre-session notes (e.g., "Feeling tired", "Working on anxiety").
6.  **Start**: Click "Start Session".

## The Session Workflow

Once started, the system enters an **Automated Mode**.

### 1. Auto-Calibration
The session begins with a 3-stage calibration phase.
*   **Instruction**: "Please squeeze the cans."
*   **Action**: "Squeeze the electrodes firmly."
*   **Release**: "Please relax."
*   **Wait**: The system waits for your reading to stabilize.
*   **Result**: The system calculates your responsiveness and sets the graph scale automatically.

### 2. The Guiding Loop
The system will now present questions one by one.
*   **Audio**: The computer voice reads the question.
*   **Text**: The question text appears in large letters on the screen.
*   **Graph**: Watch the graph trace.

### 3. Advancing the Session
**YOU are in control of the speed.**

*   **When to Advance**: Once you have fully answered the question, watch for the "Read" (graph trace dropping or reacting).
*   **How to Advance**: Press the **Spacebar** (or `Enter`) on your keyboard.
*   **Next Step**: The system will proceed to the next question immediately.

> **Tip**: Do not rush. Let the graph trace settle before pressing Spacebar.

### 4. Ending the Session
When the process is complete (all questions asked):
*   **Message**: "SESSION COMPLETE" appears on the screen.
*   **Closing**: The specific "End of Session" questions will be asked.
*   **Finish**: The recording stops automatically, and the session data is saved to `Session_Data/`.

## Controls

| Key / Action | Function |
| :--- | :--- |
| **Spacebar** | **Next Question** (Advance Step) |
| **Q** | **Initiates end of session** |
| **Stop Button** | Abort session immediately |

## Customizing Processes
Processes are defined in `processes.json`. You can create your own workflows by editing this file.
*   **Structure**: Define a list of `steps` (question keys).
*   **Loops**: Use `repeat` logic to loop questions.
*   **Assessments**: Link to lists in `assessments.json`.

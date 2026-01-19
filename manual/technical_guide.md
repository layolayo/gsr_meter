# Technical Guide & Troubleshooting 🛠️

## Signal Flow
`Sensor -> USB/ANT+ -> Python Driver -> Ring Buffer -> Analysis Engine -> UI`

## 📉 Troubleshooting Signals

### GSR Line is Flat / Grey
*   Check USB connection.
*   Ensure fingers are making good contact with electrodes.
*   Dry skin? Try a *tiny* bit of moisture (lotion/water) on the contact points.

### HRM Not Connecting
*   Ensure the ANT+ USB stick is plugged in.
*   Moisten the heart rate strap contact pads.
*   Check if another app is hogging the ANT+ stick.

### "Resource Busy" Error
*   This means the USB device is locked by a zombie process. 
*   **Fix**: Restart the application or re-plug the USB stick.

## 📊 Data Files
Session data is saved in `Session_Data/Session_YYYYMMDD_HHMMSS/`.
*   `GSR.csv`: High-frequency skin conductance data.
*   `HRM.csv`: Heart rate and variability metrics.
*   `Audio.wav`: Syncronized audio recording (if enabled).

## 🧮 Calibration Logic
The calibration wizard determines your **Drop Ratio**:
`Ratio = (Baseline - Squeeze_Drop) / Sensitivity`

*   **Target**: after a good calibration - the breath test should take up about 40% of the graph window
*   If your sensitivity is too low, the graph will be flat and unresponsive.
*   If too high, the graph will jump and fluctuate so readings will be difficult to interpret.

> **Pro Tip**: Use the **Auto-Boost** buttons (L/M/H) to dynamically resize the window if you are doing wild breathwork! 🌬️

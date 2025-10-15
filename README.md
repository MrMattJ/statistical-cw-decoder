# Statistical CW Decoder

A real-time Morse Code (CW) decoder using K-Means clustering for automatic timing calibration. This application listens to audio input and decodes Morse code using statistical analysis of signal timing patterns.

**Two versions available:**
- **Standard Decoder**: Single processor with visualization and histograms
- **Confidence-Based Decoder**: 4 parallel processors (1s/5s/10s/20s) with progressive confidence refinement ⭐ **Recommended**

## Features

### Standard Decoder (`statistical_decoder_gui.py`)

- **Real-time decoding** of Morse code from audio input
- **K-Means clustering** for automatic timing calibration (no manual calibration needed!)
- **Visual feedback** with:
  - Signal/Space duration histograms
  - Binary waveform display with color-coded classifications
  - Live decoded text output
  - Debug console
- **Adjustable parameters**:
  - Signal/Space strictness (bucket sensitivity)
  - Processing interval (0.25-20 seconds)
  - Lookback window (number of recent samples for calibration)

### Confidence-Based Decoder (`statistical_decoder_confidence.py`) ⭐

- **4 parallel processors** running at 1s, 5s, 10s, and 20s intervals
- **Progressive confidence visualization**:
  - Gray (1s) - Fast, initial decode
  - White (5s) - Medium confidence, replaces gray
  - Green (10s) - High confidence, replaces white
  - Cyan (20s) - Maximum confidence, replaces green
- **Individual controls** for each processor:
  - Separate strictness settings
  - Separate signal/space history settings
- **Simple replacement logic**: Each processor removes all text from the previous level
- **Debug console** for real-time monitoring

## Installation

### Prerequisites

- Python 3.8 or higher
- A microphone or audio input device

### Install Dependencies

```bash
pip install -r requirements.txt
```

## Usage

### Windows

**Confidence-Based Decoder (Recommended):**
```bash
launch_confidence_decoder.bat
```
Or run directly:
```bash
pythonw statistical_decoder_confidence.py
```

**Standard Decoder:**
```bash
launch_decoder.bat
```
Or run directly:
```bash
pythonw statistical_decoder_gui.py
```

### Mac/Linux

**Confidence-Based Decoder:**
```bash
python3 statistical_decoder_confidence.py
```

**Standard Decoder:**
```bash
python3 statistical_decoder_gui.py
```

## How It Works

### Standard Decoder

The decoder uses a sophisticated statistical approach:

1. **Audio Processing**: Captures audio input and extracts a binary ON/OFF envelope using hysteresis thresholding
2. **Segment Extraction**: Converts the binary signal into timed segments (ON = signal pulse, OFF = silence)
3. **K-Means Clustering**:
   - Clusters ON segments into 2 groups: Dit (short) and Dah (long)
   - Clusters OFF segments into 3 groups: Intra-character, Inter-character, and Word spaces
4. **Symbol-Based Decoding**: Processes each segment as it arrives to build and decode Morse characters in real-time

### Confidence-Based Decoder

The confidence-based decoder runs **4 independent processors in parallel** with a simple replacement strategy:

**Timeline Example:**
```
t=1s:  1s processor → "HI" (gray, quick but uncertain)
t=5s:  5s processor → Deletes ALL gray → "HELLO" (white, better accuracy)
t=10s: 10s processor → Deletes ALL white → "HELLO" (green, high accuracy)
t=20s: 20s processor → Deletes ALL green → "HELLO WORLD" (cyan, maximum accuracy)
```

**Key Advantages:**
- Immediate feedback (1s processor shows results fast)
- Progressive accuracy (text gets more reliable over time)
- Visual confidence (color indicates how certain the decode is)
- No complex time-window overlap logic - simple and robust

## Tips for Best Results

- **Processing Interval**: For best accuracy, use a longer processing interval (5-20 seconds). This gives the K-Means algorithm more data to work with.
- **Strictness**: Lower strictness values (0.05-0.5 std dev) work better for clean signals. Higher values are more forgiving but may be less accurate.
- **Signal Quality**: Works best with clean Morse code at 12-45 WPM. Reduce background noise for optimal results.

## Controls

- **Start/Stop Decoding**: Begin or stop listening to audio input
- **Clear Data**: Reset all collected timing data and decoded output
- **Signal/Space Strictness**: Adjust how tightly buckets cluster around detected patterns
- **Signal/Space History**: Number of recent samples to use for calibration
- **Processing Interval**: How often to process accumulated audio (longer = more accurate clustering)

## Requirements

See `requirements.txt` for complete list of dependencies.

## License

MIT License

## Author

Created for amateur radio (HAM) operators and Morse code enthusiasts.

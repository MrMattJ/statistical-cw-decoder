# Statistical CW Decoder

A real-time Morse Code (CW) decoder using K-Means clustering for automatic timing calibration. This application listens to audio input and decodes Morse code using statistical analysis of signal timing patterns.

## Features

- **Real-time decoding** of Morse code from audio input
- **K-Means clustering** for automatic timing calibration (no manual calibration needed!)
- **Visual feedback** with:
  - Signal/Space duration histograms
  - Binary waveform display with color-coded classifications
  - Live decoded text output
- **Adjustable parameters**:
  - Signal/Space strictness (bucket sensitivity)
  - Processing interval (0.25-20 seconds)
  - Lookback window (number of recent samples for calibration)

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

Double-click `launch_decoder.bat` or run:

```bash
pythonw statistical_decoder_gui.py
```

### Mac/Linux

```bash
python3 statistical_decoder_gui.py
```

## How It Works

The decoder uses a sophisticated statistical approach:

1. **Audio Processing**: Captures audio input and extracts a binary ON/OFF envelope using hysteresis thresholding
2. **Segment Extraction**: Converts the binary signal into timed segments (ON = signal pulse, OFF = silence)
3. **K-Means Clustering**:
   - Clusters ON segments into 2 groups: Dit (short) and Dah (long)
   - Clusters OFF segments into 3 groups: Intra-character, Inter-character, and Word spaces
4. **Symbol-Based Decoding**: Processes each segment as it arrives to build and decode Morse characters in real-time

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

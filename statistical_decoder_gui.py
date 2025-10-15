"""
Statistical CW Decoder with Histogram Visualization
Uses K-Means clustering for automatic timing calibration
"""

import os
import numpy as np
import tkinter as tk
from tkinter import ttk
from scipy.signal import hilbert, butter, filtfilt
from sklearn.cluster import KMeans
import sounddevice as sd
import threading
import matplotlib
matplotlib.use('TkAgg')
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Audio parameters
SAMPLE_RATE = 8000
HOP_LENGTH = 64
F_MIN = 600
F_MAX = 800
TARGET_CENTER_FREQ = 700

# Morse code lookup table
MORSE_TO_CHAR = {
    '.-': 'A', '-...': 'B', '-.-.': 'C', '-..': 'D', '.': 'E',
    '..-.': 'F', '--.': 'G', '....': 'H', '..': 'I', '.---': 'J',
    '-.-': 'K', '.-..': 'L', '--': 'M', '-.': 'N', '---': 'O',
    '.--.': 'P', '--.-': 'Q', '.-.': 'R', '...': 'S', '-': 'T',
    '..-': 'U', '...-': 'V', '.--': 'W', '-..-': 'X', '-.--': 'Y',
    '--..': 'Z',
    '-----': '0', '.----': '1', '..---': '2', '...--': '3',
    '....-': '4', '.....': '5', '-....': '6', '--...': '7',
    '---..': '8', '----.': '9',
    '--..--': ',', '.-.-.-': '.', '..--..': '?', '-..-.': '/'
}


def detect_peak_frequency(audio, sample_rate):
    """Detect dominant frequency using FFT"""
    fft_result = np.fft.rfft(audio)
    freqs = np.fft.rfftfreq(len(audio), 1/sample_rate)
    magnitudes = np.abs(fft_result)

    valid_mask = (freqs >= 300) & (freqs <= 1100)
    valid_freqs = freqs[valid_mask]
    valid_mags = magnitudes[valid_mask]

    if len(valid_mags) > 0:
        peak_idx = np.argmax(valid_mags)
        return valid_freqs[peak_idx]
    return TARGET_CENTER_FREQ


def frequency_shift_audio(audio, shift_hz, sample_rate):
    """Shift audio frequency"""
    t = np.arange(len(audio)) / sample_rate
    shift_signal = np.exp(2j * np.pi * shift_hz * t)
    analytic_signal = hilbert(audio)
    shifted_analytic = analytic_signal * shift_signal
    return np.real(shifted_analytic)


def extract_binary_envelope(audio, sample_rate, hop_length=64):
    """Convert audio to binary ON/OFF using hysteresis thresholding"""
    # Bandpass filter
    nyquist = sample_rate / 2
    low = F_MIN / nyquist
    high = F_MAX / nyquist
    b, a = butter(4, [low, high], btype='band')
    filtered = filtfilt(b, a, audio)

    # Get amplitude envelope
    analytic_signal = hilbert(filtered)
    envelope = np.abs(analytic_signal)
    envelope = envelope[::hop_length]

    # Normalize
    if envelope.max() > 0:
        envelope = envelope / envelope.max()

    # Hysteresis thresholding
    THRESHOLD_HIGH = 0.15
    THRESHOLD_LOW = 0.08

    binary_envelope = np.zeros(len(envelope), dtype=np.float32)
    state = 0
    for i in range(len(envelope)):
        if state == 0:  # OFF
            if envelope[i] > THRESHOLD_HIGH:
                state = 1
        else:  # ON
            if envelope[i] < THRESHOLD_LOW:
                state = 0
        binary_envelope[i] = float(state)

    return binary_envelope


def extract_segments(binary_signal, frame_rate_hz=125):
    """
    Convert binary signal to run-length encoded segments
    Returns: [(state, duration_ms), ...]
    state: 1=ON (tone), 0=OFF (silence)
    """
    segments = []
    if len(binary_signal) == 0:
        return segments

    current_state = binary_signal[0]
    current_length = 1

    for i in range(1, len(binary_signal)):
        if binary_signal[i] == current_state:
            current_length += 1
        else:
            # State changed - save segment
            duration_ms = (current_length / frame_rate_hz) * 1000
            segments.append((int(current_state), duration_ms))
            current_state = binary_signal[i]
            current_length = 1

    # Save final segment
    duration_ms = (current_length / frame_rate_hz) * 1000
    segments.append((int(current_state), duration_ms))

    return segments


class StatisticalBuckets:
    """Manages statistical buckets using K-Means clustering"""

    def __init__(self, n_buckets, std_devs=2.0):
        self.n_buckets = n_buckets
        self.std_devs = std_devs
        self.centers = None
        self.thresholds = None

    def fit(self, durations):
        """Fit buckets using K-Means clustering"""
        if len(durations) < self.n_buckets * 2:
            return False

        durations_array = np.array(durations).reshape(-1, 1)

        # Use K-Means to find cluster centers
        kmeans = KMeans(n_clusters=self.n_buckets, random_state=42, n_init=10).fit(durations_array)
        self.centers = sorted(kmeans.cluster_centers_.flatten())

        # Calculate thresholds as midpoints between adjacent centers
        self.thresholds = []
        for i in range(len(self.centers) - 1):
            threshold = (self.centers[i] + self.centers[i + 1]) / 2
            self.thresholds.append(threshold)

        return True

    def classify(self, duration):
        """
        Classify a duration into a bucket using thresholds
        Returns: bucket_index (0, 1, 2...) or None
        """
        if self.centers is None or self.thresholds is None:
            return None

        # Compare against thresholds to determine bucket
        for i, threshold in enumerate(self.thresholds):
            if duration < threshold:
                return i

        # If greater than all thresholds, it's in the last bucket
        return len(self.centers) - 1


class StatisticalDecoderGUI:
    def __init__(self, root):
        self.root = root

        # Live decoding state
        self.is_decoding = False
        self.audio_stream = None
        self.audio_buffer = []
        self.processing_lock = threading.Lock()
        self.pending_segment = None

        # Collected durations (raw data)
        self.signal_durations = []  # All ON durations
        self.space_durations = []   # All OFF durations

        # Real-time symbol-based decoding state
        self.current_morse_pattern = []  # Accumulating dits/dahs for current character
        self.last_segment_state = None   # Track state transitions

        # Statistical buckets
        self.signal_buckets = StatisticalBuckets(n_buckets=2, std_devs=2.0)  # Dit, Dah
        self.space_buckets = StatisticalBuckets(n_buckets=3, std_devs=2.0)   # Intra, Inter, Word

        # Decoded output
        self.decoded_text_output = ""

        # Waveform history for plotting
        self.waveform_history = []  # Binary signal history
        self.classification_history = []  # List of (start_frame, end_frame, classification)
        self.max_waveform_frames = 1000  # Show last 1000 frames (~8 seconds at 125 Hz)

        # Standard deviation controls (separate for signals and spaces)
        self.signal_std_devs = tk.DoubleVar(value=2.0)  # n standard deviations for signal buckets
        self.space_std_devs = tk.DoubleVar(value=2.0)   # n standard deviations for space buckets

        # Lookback window controls (how many recent samples to use for buckets)
        self.signal_lookback = tk.IntVar(value=100)  # Use last X signals for calibration
        self.space_lookback = tk.IntVar(value=100)   # Use last Y spaces for calibration

        # Processing interval control
        self.processing_interval = tk.DoubleVar(value=2.0)  # Process every N seconds

        root.title("Statistical CW Decoder")
        root.geometry("1400x1100")
        root.configure(bg='#2b2b2b')

        self.setup_ui()

    def setup_ui(self):
        # Title
        title_label = tk.Label(
            self.root,
            text="Statistical CW Decoder - Histogram Analysis",
            font=('Consolas', 16, 'bold'),
            bg='#2b2b2b',
            fg='#00ff00'
        )
        title_label.pack(pady=10)

        # Top container for controls and output
        top_container = tk.Frame(self.root, bg='#2b2b2b')
        top_container.pack(fill=tk.X, padx=20, pady=10)

        # Control panel (left side)
        control_frame = tk.Frame(top_container, bg='#1e1e1e', relief=tk.RIDGE, borderwidth=2)
        control_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))

        # Decode button
        button_frame = tk.Frame(control_frame, bg='#1e1e1e')
        button_frame.pack(pady=10)

        self.decode_button = tk.Button(
            button_frame,
            text="▶ Start Decoding",
            command=self.toggle_decoding,
            bg='#4CAF50',
            fg='white',
            font=('Consolas', 11, 'bold'),
            width=18
        )
        self.decode_button.pack(side=tk.LEFT, padx=10)

        tk.Button(
            button_frame,
            text="🗑️ Clear Data",
            command=self.clear_data,
            bg='#FF9800',
            fg='white',
            font=('Consolas', 10, 'bold'),
            width=15
        ).pack(side=tk.LEFT, padx=10)

        # Signal strictness slider
        signal_slider_frame = tk.Frame(control_frame, bg='#1e1e1e')
        signal_slider_frame.pack(fill=tk.X, padx=20, pady=(10, 5))

        tk.Label(
            signal_slider_frame,
            text="Signal Strictness (± std devs):",
            font=('Consolas', 10),
            bg='#1e1e1e',
            fg='white',
            width=30,
            anchor='w'
        ).pack(side=tk.LEFT)

        self.signal_std_dev_slider = tk.Scale(
            signal_slider_frame,
            from_=0.05,
            to=4.0,
            resolution=0.05,
            orient=tk.HORIZONTAL,
            variable=self.signal_std_devs,
            bg='#2b2b2b',
            fg='white',
            highlightthickness=0,
            length=300,
            command=self.update_signal_strictness
        )
        self.signal_std_dev_slider.pack(side=tk.LEFT, padx=10)

        self.signal_std_dev_label = tk.Label(
            signal_slider_frame,
            text="± 2.0 σ",
            font=('Consolas', 10, 'bold'),
            bg='#1e1e1e',
            fg='#00ff00',
            width=15,
            anchor='w'
        )
        self.signal_std_dev_label.pack(side=tk.LEFT, padx=10)

        # Space strictness slider
        space_slider_frame = tk.Frame(control_frame, bg='#1e1e1e')
        space_slider_frame.pack(fill=tk.X, padx=20, pady=(5, 10))

        tk.Label(
            space_slider_frame,
            text="Space Strictness (± std devs):",
            font=('Consolas', 10),
            bg='#1e1e1e',
            fg='white',
            width=30,
            anchor='w'
        ).pack(side=tk.LEFT)

        self.space_std_dev_slider = tk.Scale(
            space_slider_frame,
            from_=0.05,
            to=4.0,
            resolution=0.05,
            orient=tk.HORIZONTAL,
            variable=self.space_std_devs,
            bg='#2b2b2b',
            fg='white',
            highlightthickness=0,
            length=300,
            command=self.update_space_strictness
        )
        self.space_std_dev_slider.pack(side=tk.LEFT, padx=10)

        self.space_std_dev_label = tk.Label(
            space_slider_frame,
            text="± 2.0 σ",
            font=('Consolas', 10, 'bold'),
            bg='#1e1e1e',
            fg='#00ff00',
            width=15,
            anchor='w'
        )
        self.space_std_dev_label.pack(side=tk.LEFT, padx=10)

        # Lookback window sliders
        lookback_label_frame = tk.Frame(control_frame, bg='#1e1e1e')
        lookback_label_frame.pack(fill=tk.X, padx=20, pady=(10, 0))

        tk.Label(
            lookback_label_frame,
            text="Bucket Calibration Window (Recent Samples Only):",
            font=('Consolas', 10, 'bold'),
            bg='#1e1e1e',
            fg='#FFC107'
        ).pack(anchor='w')

        # Signal lookback slider
        signal_lookback_frame = tk.Frame(control_frame, bg='#1e1e1e')
        signal_lookback_frame.pack(fill=tk.X, padx=20, pady=(5, 5))

        tk.Label(
            signal_lookback_frame,
            text="Signal History (samples):",
            font=('Consolas', 10),
            bg='#1e1e1e',
            fg='white',
            width=30,
            anchor='w'
        ).pack(side=tk.LEFT)

        self.signal_lookback_slider = tk.Scale(
            signal_lookback_frame,
            from_=20,
            to=500,
            resolution=10,
            orient=tk.HORIZONTAL,
            variable=self.signal_lookback,
            bg='#2b2b2b',
            fg='white',
            highlightthickness=0,
            length=300,
            command=self.update_signal_lookback
        )
        self.signal_lookback_slider.pack(side=tk.LEFT, padx=10)

        self.signal_lookback_label = tk.Label(
            signal_lookback_frame,
            text="100 samples",
            font=('Consolas', 10, 'bold'),
            bg='#1e1e1e',
            fg='#00ff00',
            width=15,
            anchor='w'
        )
        self.signal_lookback_label.pack(side=tk.LEFT, padx=10)

        # Space lookback slider
        space_lookback_frame = tk.Frame(control_frame, bg='#1e1e1e')
        space_lookback_frame.pack(fill=tk.X, padx=20, pady=(5, 10))

        tk.Label(
            space_lookback_frame,
            text="Space History (samples):",
            font=('Consolas', 10),
            bg='#1e1e1e',
            fg='white',
            width=30,
            anchor='w'
        ).pack(side=tk.LEFT)

        self.space_lookback_slider = tk.Scale(
            space_lookback_frame,
            from_=20,
            to=500,
            resolution=10,
            orient=tk.HORIZONTAL,
            variable=self.space_lookback,
            bg='#2b2b2b',
            fg='white',
            highlightthickness=0,
            length=300,
            command=self.update_space_lookback
        )
        self.space_lookback_slider.pack(side=tk.LEFT, padx=10)

        self.space_lookback_label = tk.Label(
            space_lookback_frame,
            text="100 samples",
            font=('Consolas', 10, 'bold'),
            bg='#1e1e1e',
            fg='#00ff00',
            width=15,
            anchor='w'
        )
        self.space_lookback_label.pack(side=tk.LEFT, padx=10)

        # Processing interval slider
        interval_label_frame = tk.Frame(control_frame, bg='#1e1e1e')
        interval_label_frame.pack(fill=tk.X, padx=20, pady=(10, 0))

        tk.Label(
            interval_label_frame,
            text="Processing Cadence:",
            font=('Consolas', 10, 'bold'),
            bg='#1e1e1e',
            fg='#FFC107'
        ).pack(anchor='w')

        interval_slider_frame = tk.Frame(control_frame, bg='#1e1e1e')
        interval_slider_frame.pack(fill=tk.X, padx=20, pady=(5, 10))

        tk.Label(
            interval_slider_frame,
            text="Processing Interval (seconds):",
            font=('Consolas', 10),
            bg='#1e1e1e',
            fg='white',
            width=30,
            anchor='w'
        ).pack(side=tk.LEFT)

        self.interval_slider = tk.Scale(
            interval_slider_frame,
            from_=0.25,
            to=20.0,
            resolution=0.25,
            orient=tk.HORIZONTAL,
            variable=self.processing_interval,
            bg='#2b2b2b',
            fg='white',
            highlightthickness=0,
            length=300,
            command=self.update_processing_interval
        )
        self.interval_slider.pack(side=tk.LEFT, padx=10)

        self.interval_label = tk.Label(
            interval_slider_frame,
            text="2.0 seconds",
            font=('Consolas', 10, 'bold'),
            bg='#1e1e1e',
            fg='#00ff00',
            width=15,
            anchor='w'
        )
        self.interval_label.pack(side=tk.LEFT, padx=10)

        # Decoded output (right side)
        output_frame = tk.Frame(top_container, bg='#1e1e1e', relief=tk.RIDGE, borderwidth=2)
        output_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=False)

        tk.Label(
            output_frame,
            text="Decoded Output",
            font=('Consolas', 11, 'bold'),
            bg='#1e1e1e',
            fg='#FFC107'
        ).pack(pady=5)

        self.decoded_text = tk.Text(
            output_frame,
            height=20,
            width=40,
            bg='#2b2b2b',
            fg='#00ff00',
            font=('Consolas', 10),
            wrap=tk.WORD
        )
        self.decoded_text.pack(padx=10, pady=(0, 10), fill=tk.BOTH, expand=True)

        # Status display
        status_frame = tk.Frame(self.root, bg='#1e1e1e', relief=tk.RIDGE, borderwidth=2)
        status_frame.pack(fill=tk.X, padx=20, pady=(0, 10))

        self.status_label = tk.Label(
            status_frame,
            text="Status: Ready",
            font=('Consolas', 10),
            bg='#1e1e1e',
            fg='#FFC107',
            anchor='w'
        )
        self.status_label.pack(fill=tk.X, padx=10, pady=5)

        # Histogram plots (side by side)
        plot_frame = tk.Frame(self.root, bg='#2b2b2b')
        plot_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)

        # Left plot: Signal durations
        left_plot_frame = tk.Frame(plot_frame, bg='#1e1e1e', relief=tk.RIDGE, borderwidth=2)
        left_plot_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))

        tk.Label(
            left_plot_frame,
            text="Signal Durations (Dit / Dah)",
            font=('Consolas', 11, 'bold'),
            bg='#1e1e1e',
            fg='#FFC107'
        ).pack(pady=5)

        self.signal_fig = Figure(figsize=(6, 4), facecolor='#2b2b2b')
        self.signal_fig.tight_layout(pad=2.0)
        self.signal_ax = self.signal_fig.add_subplot(111, facecolor='#1e1e1e')
        self.signal_ax.set_xlabel('Duration (ms)', color='white')
        self.signal_ax.set_ylabel('Count', color='white')
        self.signal_ax.tick_params(colors='white')
        for spine in self.signal_ax.spines.values():
            spine.set_color('white')

        self.signal_canvas = FigureCanvasTkAgg(self.signal_fig, master=left_plot_frame)
        self.signal_canvas.draw()
        self.signal_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Right plot: Space durations
        right_plot_frame = tk.Frame(plot_frame, bg='#1e1e1e', relief=tk.RIDGE, borderwidth=2)
        right_plot_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(5, 0))

        tk.Label(
            right_plot_frame,
            text="Space Durations (Intra / Inter / Word)",
            font=('Consolas', 11, 'bold'),
            bg='#1e1e1e',
            fg='#FFC107'
        ).pack(pady=5)

        self.space_fig = Figure(figsize=(6, 4), facecolor='#2b2b2b')
        self.space_fig.tight_layout(pad=2.0)
        self.space_ax = self.space_fig.add_subplot(111, facecolor='#1e1e1e')
        self.space_ax.set_xlabel('Duration (ms)', color='white')
        self.space_ax.set_ylabel('Count', color='white')
        self.space_ax.tick_params(colors='white')
        for spine in self.space_ax.spines.values():
            spine.set_color('white')

        self.space_canvas = FigureCanvasTkAgg(self.space_fig, master=right_plot_frame)
        self.space_canvas.draw()
        self.space_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Waveform plot
        waveform_frame = tk.Frame(self.root, bg='#1e1e1e', relief=tk.RIDGE, borderwidth=2)
        waveform_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)

        tk.Label(
            waveform_frame,
            text="Binary Waveform (ON/OFF Signal)",
            font=('Consolas', 11, 'bold'),
            bg='#1e1e1e',
            fg='#FFC107'
        ).pack(pady=5)

        self.waveform_fig = Figure(figsize=(10, 3), facecolor='#2b2b2b')
        self.waveform_fig.tight_layout(pad=2.0)
        self.waveform_ax = self.waveform_fig.add_subplot(111, facecolor='#1e1e1e')
        self.waveform_ax.set_xlabel('Time (frames, ~8ms each)', color='white')
        self.waveform_ax.set_ylabel('Signal', color='white')
        self.waveform_ax.tick_params(colors='white')
        for spine in self.waveform_ax.spines.values():
            spine.set_color('white')

        self.waveform_canvas = FigureCanvasTkAgg(self.waveform_fig, master=waveform_frame)
        self.waveform_canvas.draw()
        self.waveform_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 5))

    def update_signal_strictness(self, value):
        """Update signal bucket strictness when slider changes"""
        n = float(value)
        self.signal_std_dev_label.config(text=f"± {n:.1f} σ")
        self.signal_buckets.std_devs = n
        # Redraw histograms with new boundaries
        self.update_histograms()

    def update_space_strictness(self, value):
        """Update space bucket strictness when slider changes"""
        n = float(value)
        self.space_std_dev_label.config(text=f"± {n:.1f} σ")
        self.space_buckets.std_devs = n
        # Redraw histograms with new boundaries
        self.update_histograms()

    def update_signal_lookback(self, value):
        """Update signal lookback window when slider changes"""
        n = int(float(value))
        self.signal_lookback_label.config(text=f"{n} samples")

    def update_space_lookback(self, value):
        """Update space lookback window when slider changes"""
        n = int(float(value))
        self.space_lookback_label.config(text=f"{n} samples")

    def update_processing_interval(self, value):
        """Update processing interval when slider changes"""
        n = float(value)
        self.interval_label.config(text=f"{n:.2f} seconds")

    def toggle_decoding(self):
        """Toggle continuous decoding on/off"""
        if self.is_decoding:
            self.stop_decoding()
        else:
            self.start_decoding()

    def start_decoding(self):
        """Start continuous real-time decoding"""
        if self.is_decoding:
            return

        self.is_decoding = True
        self.decode_button.config(text="⏹ Stop Decoding", bg='#FF0000')
        self.status_label.config(text="Status: Listening for CW signal...")

        # Reset state
        self.audio_buffer = []
        self.pending_segment = None
        self.current_morse_pattern = []

        # Start audio stream
        try:
            self.audio_stream = sd.InputStream(
                samplerate=SAMPLE_RATE,
                channels=1,
                dtype='float32',
                blocksize=4000,
                callback=self.audio_callback
            )
            self.audio_stream.start()

        except Exception as e:
            print(f"Error starting audio stream: {e}")
            self.stop_decoding()
            self.status_label.config(text=f"ERROR: {e}")

    def stop_decoding(self):
        """Stop continuous decoding"""
        if not self.is_decoding:
            return

        self.is_decoding = False
        self.decode_button.config(text="▶ Start Decoding", bg='#4CAF50')

        if self.audio_stream:
            self.audio_stream.stop()
            self.audio_stream.close()
            self.audio_stream = None

        self.status_label.config(text="Status: Stopped")

    def audio_callback(self, indata, frames, time_info, status):
        """Callback for continuous audio streaming"""
        if status:
            print(f"Audio status: {status}")

        audio_chunk = indata[:, 0].copy()
        self.audio_buffer.extend(audio_chunk)

        # Process at user-defined interval
        interval = self.processing_interval.get()
        if len(self.audio_buffer) >= SAMPLE_RATE * interval:
            threading.Thread(target=self.process_audio_buffer, daemon=True).start()

    def process_audio_buffer(self):
        """Process accumulated audio buffer"""
        with self.processing_lock:
            if not self.is_decoding or len(self.audio_buffer) == 0:
                return

            audio = np.array(self.audio_buffer)

            # Keep last 0.5 seconds for continuity
            if len(audio) > SAMPLE_RATE * 1.0:
                keep_samples = int(SAMPLE_RATE * 0.5)
                self.audio_buffer = self.audio_buffer[-keep_samples:]
            else:
                self.audio_buffer = []

        # Process audio
        try:
            # Frequency centering
            peak_freq = detect_peak_frequency(audio, SAMPLE_RATE)
            shift_hz = TARGET_CENTER_FREQ - peak_freq
            audio = frequency_shift_audio(audio, shift_hz, SAMPLE_RATE)

            # Convert to binary
            binary_signal = extract_binary_envelope(audio, SAMPLE_RATE, HOP_LENGTH)

            # Check if we got any signal
            on_percent = binary_signal.sum() / len(binary_signal)
            if on_percent < 0.01:
                return

            # Extract segments
            segments = extract_segments(binary_signal, frame_rate_hz=125)

            if not segments:
                return

            # Handle pending segment from previous buffer
            if self.pending_segment is not None:
                pending_state, pending_duration = self.pending_segment
                first_state, first_duration = segments[0]

                if pending_state == first_state:
                    segments[0] = (first_state, pending_duration + first_duration)
                else:
                    segments.insert(0, self.pending_segment)

                self.pending_segment = None

            # Check if last segment is incomplete
            last_state, last_duration = segments[-1]
            final_binary_state = int(binary_signal[-1])

            if last_state == final_binary_state:
                self.pending_segment = segments[-1]
                segments = segments[:-1]

            # Collect durations
            for state, duration in segments:
                if state == 1:  # Signal (ON)
                    self.signal_durations.append(duration)
                else:  # Space (OFF)
                    self.space_durations.append(duration)

            # Keep only the lookback window of most recent samples
            signal_lookback = self.signal_lookback.get()
            space_lookback = self.space_lookback.get()

            if len(self.signal_durations) > signal_lookback:
                self.signal_durations = self.signal_durations[-signal_lookback:]
            if len(self.space_durations) > space_lookback:
                self.space_durations = self.space_durations[-space_lookback:]

            # Update buckets if we have enough data
            min_signals = 10
            min_spaces = 10

            # Symbol-based parsing: process each segment as it arrives
            segment_classifications = []
            decoded_chars = []

            if len(self.signal_durations) >= min_signals and len(self.space_durations) >= min_spaces:
                # Fit buckets using only recent samples (already trimmed above)
                signal_fitted = self.signal_buckets.fit(self.signal_durations)
                space_fitted = self.space_buckets.fit(self.space_durations)

                if signal_fitted and space_fitted:
                    # DEBUG: Print cluster centers
                    print(f"Signal centers (Dit/Dah): {self.signal_buckets.centers}")
                    print(f"Space centers (Intra/Inter/Word): {self.space_buckets.centers}")

                    # Process each segment symbol-by-symbol
                    for state, duration in segments:
                        if state == 1:  # Signal (ON) - Dit or Dah
                            bucket = self.signal_buckets.classify(duration)
                            print(f"ON segment: {duration:.1f}ms → bucket {bucket}")
                            if bucket == 0:  # Dit
                                segment_classifications.append('Dit')
                                self.current_morse_pattern.append('.')
                            elif bucket == 1:  # Dah
                                segment_classifications.append('Dah')
                                self.current_morse_pattern.append('-')
                            else:
                                segment_classifications.append('Signal')  # Outside buckets - ignore

                        else:  # Space (OFF)
                            bucket = self.space_buckets.classify(duration)
                            print(f"OFF segment: {duration:.1f}ms → bucket {bucket}")

                            if bucket == 0:  # Intra-character space
                                segment_classifications.append('Intra')
                                # Continue building current character

                            elif bucket == 1:  # Inter-character space - CHARACTER BOUNDARY!
                                segment_classifications.append('Inter')
                                # Decode current character immediately
                                if self.current_morse_pattern:
                                    pattern = ''.join(self.current_morse_pattern)
                                    char = MORSE_TO_CHAR.get(pattern, '?')
                                    decoded_chars.append(char)
                                    self.current_morse_pattern = []

                            elif bucket == 2:  # Word space - WORD BOUNDARY!
                                segment_classifications.append('Word')
                                # Decode current character + add word space
                                if self.current_morse_pattern:
                                    pattern = ''.join(self.current_morse_pattern)
                                    char = MORSE_TO_CHAR.get(pattern, '?')
                                    decoded_chars.append(char)
                                    self.current_morse_pattern = []
                                decoded_chars.append(' ')

                            else:
                                segment_classifications.append('Space')  # Outside buckets - ignore

                    # Output decoded characters immediately
                    if decoded_chars:
                        decoded_text = ''.join(decoded_chars)
                        self.decoded_text_output += decoded_text
                        self.root.after(0, lambda txt=decoded_text: self.update_display(txt))

            # Update waveform with classifications
            self.root.after(0, lambda: self.update_waveform(binary_signal, segments, segment_classifications))

            # Update histograms
            self.root.after(0, self.update_histograms)

            # Update status
            self.root.after(0, lambda: self.status_label.config(
                text=f"Status: {len(self.signal_durations)} signals, {len(self.space_durations)} spaces collected"
            ))

        except Exception as e:
            print(f"Processing error: {e}")
            import traceback
            traceback.print_exc()

    def decode_segments(self, segments):
        """Decode segments using statistical buckets"""
        decoded_chars = []
        current_morse = []

        for state, duration in segments:
            if state == 1:  # Signal (ON)
                bucket = self.signal_buckets.classify(duration)
                if bucket is None:
                    continue  # Ignore - outside bucket boundaries

                if bucket == 0:  # Dit (shorter signal)
                    current_morse.append('.')
                else:  # Dah (longer signal)
                    current_morse.append('-')

            else:  # Space (OFF)
                bucket = self.space_buckets.classify(duration)
                if bucket is None:
                    continue  # Ignore - outside bucket boundaries

                if bucket == 0:  # Intra-character (shortest space)
                    pass  # Continue building current character
                elif bucket == 1:  # Inter-character (medium space)
                    if current_morse:
                        pattern = ''.join(current_morse)
                        char = MORSE_TO_CHAR.get(pattern, '?')
                        decoded_chars.append(char)
                        current_morse = []
                else:  # Word space (longest space)
                    if current_morse:
                        pattern = ''.join(current_morse)
                        char = MORSE_TO_CHAR.get(pattern, '?')
                        decoded_chars.append(char)
                        current_morse = []
                    decoded_chars.append(' ')

        # Final character
        if current_morse:
            pattern = ''.join(current_morse)
            char = MORSE_TO_CHAR.get(pattern, '?')
            decoded_chars.append(char)

        return ''.join(decoded_chars)

    def update_display(self, new_text):
        """Update decoded text display"""
        self.decoded_text.delete('1.0', tk.END)
        self.decoded_text.insert('1.0', self.decoded_text_output)
        self.decoded_text.see('end')

    def update_waveform(self, binary_signal, segments, classifications):
        """Update binary waveform plot with color-coded classifications"""
        try:
            # Add new binary signal to history
            self.waveform_history.extend(binary_signal)

            # Keep only last max_waveform_frames
            if len(self.waveform_history) > self.max_waveform_frames:
                overflow = len(self.waveform_history) - self.max_waveform_frames
                self.waveform_history = self.waveform_history[-self.max_waveform_frames:]

                # Adjust classification history frame indices
                new_classifications = []
                for start, end, cls in self.classification_history:
                    new_start = start - overflow
                    new_end = end - overflow
                    if new_end > 0:  # Keep if any part is still visible
                        new_classifications.append((max(0, new_start), new_end, cls))
                self.classification_history = new_classifications

            # Add new classifications to history (skip generic 'Signal' and 'Space')
            # Reconstruct actual frame positions from binary_signal to avoid rounding errors
            current_frame = len(self.waveform_history) - len(binary_signal)
            classification_idx = 0

            if len(binary_signal) > 0 and len(classifications) > 0:
                current_state = binary_signal[0]
                segment_start_frame = 0

                for i in range(1, len(binary_signal)):
                    if binary_signal[i] != current_state:
                        # State changed - this segment ends here
                        if classification_idx < len(classifications):
                            cls = classifications[classification_idx]
                            if cls not in ['Signal', 'Space']:
                                abs_start = current_frame + segment_start_frame
                                abs_end = current_frame + i
                                self.classification_history.append((abs_start, abs_end, cls))
                            classification_idx += 1

                        # Start new segment
                        current_state = binary_signal[i]
                        segment_start_frame = i

                # Handle final segment
                if classification_idx < len(classifications):
                    cls = classifications[classification_idx]
                    if cls not in ['Signal', 'Space']:
                        abs_start = current_frame + segment_start_frame
                        abs_end = current_frame + len(binary_signal)
                        self.classification_history.append((abs_start, abs_end, cls))

            # Clear and redraw plot
            self.waveform_ax.clear()

            # Plot waveform
            x = np.arange(len(self.waveform_history))
            self.waveform_ax.plot(x, self.waveform_history, color='#00ff00', linewidth=1, label='Signal')

            # Add shaded regions for classifications
            symbol_colors = {
                'Dit': ('#00ff00', 0.3),      # Green
                'Dah': ('#00ffff', 0.3),      # Cyan
                'Intra': ('#444444', 0.5),    # Dark gray (short space)
                'Inter': ('#ff8800', 0.4),    # Orange (medium space)
                'Word': ('#ffff00', 0.5),     # Yellow (long space)
                'Signal': ('#888888', 0.3),   # Gray (unclassified signal)
                'Space': ('#333333', 0.4)     # Dark gray (unclassified space)
            }

            # Track which labels we've added (for legend)
            added_labels = set()

            for start, end, cls in self.classification_history:
                if 0 <= start < len(self.waveform_history):
                    color, alpha = symbol_colors.get(cls, ('#ffffff', 0.3))
                    label = cls if cls not in added_labels else None
                    if label:
                        added_labels.add(cls)
                    self.waveform_ax.axvspan(start, end, alpha=alpha, color=color, label=label)

            # Style plot
            self.waveform_ax.set_xlim(0, self.max_waveform_frames)
            self.waveform_ax.set_ylim(-0.1, 1.1)
            self.waveform_ax.set_xlabel('Time (frames, ~8ms each)', color='white')
            self.waveform_ax.set_ylabel('Signal (ON/OFF)', color='white')
            self.waveform_ax.tick_params(colors='white')
            for spine in self.waveform_ax.spines.values():
                spine.set_color('white')
            self.waveform_ax.grid(True, alpha=0.2, color='white')

            # Add legend if we have classifications
            if added_labels:
                legend = self.waveform_ax.legend(loc='upper right', facecolor='#2b2b2b',
                                                edgecolor='white', framealpha=0.8)
                for text in legend.get_texts():
                    text.set_color('white')

            # Redraw canvas
            self.waveform_canvas.draw()

        except Exception as e:
            print(f"Error updating waveform: {e}")
            import traceback
            traceback.print_exc()

    def update_histograms(self):
        """Update histogram plots with K-Means cluster centers and thresholds"""
        # Signal histogram
        self.signal_ax.clear()

        if len(self.signal_durations) > 0:
            # Plot histogram
            bins = np.linspace(0, max(self.signal_durations) + 10, 50)
            self.signal_ax.hist(self.signal_durations, bins=bins, color='#00ff00', alpha=0.7, edgecolor='white')

            # Plot K-Means centers and thresholds if fitted
            if self.signal_buckets.centers is not None:
                colors = ['#00ffff', '#ff00ff']  # Cyan for Dit, Magenta for Dah
                labels = ['Dit', 'Dah']

                for i, center in enumerate(self.signal_buckets.centers):
                    # Plot center line
                    self.signal_ax.axvline(center, color=colors[i], linewidth=2, label=f'{labels[i]} ({center:.1f}ms)')

                # Plot threshold (midpoint between Dit and Dah)
                if self.signal_buckets.thresholds:
                    threshold = self.signal_buckets.thresholds[0]
                    self.signal_ax.axvline(threshold, color='white', linewidth=2, linestyle='--', label=f'Threshold ({threshold:.1f}ms)')

                self.signal_ax.legend(facecolor='#2b2b2b', edgecolor='white', labelcolor='white')

        self.signal_ax.set_xlabel('Duration (ms)', color='white')
        self.signal_ax.set_ylabel('Count', color='white')
        self.signal_ax.tick_params(colors='white')
        for spine in self.signal_ax.spines.values():
            spine.set_color('white')
        self.signal_ax.grid(True, alpha=0.2, color='white')
        self.signal_canvas.draw()

        # Space histogram
        self.space_ax.clear()

        if len(self.space_durations) > 0:
            # Plot histogram
            bins = np.linspace(0, max(self.space_durations) + 10, 50)
            self.space_ax.hist(self.space_durations, bins=bins, color='#ffff00', alpha=0.7, edgecolor='white')

            # Plot K-Means centers and thresholds if fitted
            if self.space_buckets.centers is not None:
                colors = ['#00ffff', '#ff8800', '#ff0000']  # Cyan, Orange, Red
                labels = ['Intra', 'Inter', 'Word']

                for i, center in enumerate(self.space_buckets.centers):
                    # Plot center line
                    self.space_ax.axvline(center, color=colors[i], linewidth=2, label=f'{labels[i]} ({center:.1f}ms)')

                # Plot thresholds (midpoints between clusters)
                if self.space_buckets.thresholds:
                    for i, threshold in enumerate(self.space_buckets.thresholds):
                        self.space_ax.axvline(threshold, color='white', linewidth=1, linestyle='--', alpha=0.7)

                self.space_ax.legend(facecolor='#2b2b2b', edgecolor='white', labelcolor='white')

        self.space_ax.set_xlabel('Duration (ms)', color='white')
        self.space_ax.set_ylabel('Count', color='white')
        self.space_ax.tick_params(colors='white')
        for spine in self.space_ax.spines.values():
            spine.set_color('white')
        self.space_ax.grid(True, alpha=0.2, color='white')
        self.space_canvas.draw()

    def clear_data(self):
        """Clear all collected data and reset"""
        with self.processing_lock:
            self.signal_durations = []
            self.space_durations = []
            self.signal_buckets = StatisticalBuckets(n_buckets=2, std_devs=self.signal_std_devs.get())
            self.space_buckets = StatisticalBuckets(n_buckets=3, std_devs=self.space_std_devs.get())
            self.decoded_text_output = ""
            self.waveform_history = []
            self.classification_history = []
            self.current_morse_pattern = []

        self.decoded_text.delete('1.0', tk.END)
        self.status_label.config(text="Status: Data cleared - ready for new signal")

        # Clear waveform
        self.waveform_ax.clear()
        self.waveform_ax.set_xlabel('Time (frames, ~8ms each)', color='white')
        self.waveform_ax.set_ylabel('Signal (ON/OFF)', color='white')
        self.waveform_ax.tick_params(colors='white')
        for spine in self.waveform_ax.spines.values():
            spine.set_color('white')
        self.waveform_ax.text(0.5, 0.5, 'Waiting for signal...',
                             transform=self.waveform_ax.transAxes, ha='center', va='center',
                             color='white', fontsize=14)
        self.waveform_canvas.draw()

        self.update_histograms()


if __name__ == '__main__':
    root = tk.Tk()
    app = StatisticalDecoderGUI(root)
    root.mainloop()

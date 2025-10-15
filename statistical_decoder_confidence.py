"""
Statistical CW Decoder with Confidence-Based Output
Uses 1s, 5s, and 10s processors with progressive confidence visualization
"""

import os
import numpy as np
import tkinter as tk
from tkinter import ttk
from scipy.signal import hilbert, butter, filtfilt
from sklearn.cluster import KMeans
import sounddevice as sd
import threading
import time

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


class DecodingProcessor:
    """Single processor for a specific time window"""

    def __init__(self, name, interval_seconds, signal_lookback=20, space_lookback=20, strictness=0.05):
        self.name = name
        self.interval_seconds = interval_seconds
        self.signal_lookback = signal_lookback
        self.space_lookback = space_lookback
        self.strictness = strictness

        # Timing data
        self.signal_durations = []
        self.space_durations = []

        # K-Means models
        self.signal_buckets = StatisticalBuckets(n_buckets=2, std_devs=strictness)
        self.space_buckets = StatisticalBuckets(n_buckets=3, std_devs=strictness)

        # Morse pattern building
        self.current_morse_pattern = []

    def update_strictness(self, strictness):
        """Update strictness for both buckets"""
        self.strictness = strictness
        self.signal_buckets.std_devs = strictness
        self.space_buckets.std_devs = strictness

    def update_lookback(self, signal_lookback, space_lookback):
        """Update lookback windows"""
        self.signal_lookback = signal_lookback
        self.space_lookback = space_lookback

    def process_segments(self, segments):
        """Process segments and return decoded text"""
        # Collect durations
        for state, duration in segments:
            if state == 1:  # Signal (ON)
                self.signal_durations.append(duration)
            else:  # Space (OFF)
                self.space_durations.append(duration)

        # Trim to lookback windows (separate for signal and space)
        if len(self.signal_durations) > self.signal_lookback:
            self.signal_durations = self.signal_durations[-self.signal_lookback:]
        if len(self.space_durations) > self.space_lookback:
            self.space_durations = self.space_durations[-self.space_lookback:]

        # Need minimum data
        if len(self.signal_durations) < 10 or len(self.space_durations) < 10:
            return ""

        # Fit K-Means models
        signal_fitted = self.signal_buckets.fit(self.signal_durations)
        space_fitted = self.space_buckets.fit(self.space_durations)

        if not signal_fitted or not space_fitted:
            return ""

        # Decode segments
        decoded_chars = []
        for state, duration in segments:
            if state == 1:  # Signal (ON)
                bucket = self.signal_buckets.classify(duration)
                if bucket == 0:  # Dit
                    self.current_morse_pattern.append('.')
                elif bucket == 1:  # Dah
                    self.current_morse_pattern.append('-')
            else:  # Space (OFF)
                bucket = self.space_buckets.classify(duration)
                if bucket == 0:  # Intra-character space
                    pass  # Continue building character
                elif bucket == 1:  # Inter-character space
                    if self.current_morse_pattern:
                        pattern = ''.join(self.current_morse_pattern)
                        char = MORSE_TO_CHAR.get(pattern, '?')
                        decoded_chars.append(char)
                        self.current_morse_pattern = []
                elif bucket == 2:  # Word space
                    if self.current_morse_pattern:
                        pattern = ''.join(self.current_morse_pattern)
                        char = MORSE_TO_CHAR.get(pattern, '?')
                        decoded_chars.append(char)
                        self.current_morse_pattern = []
                    decoded_chars.append(' ')

        return ''.join(decoded_chars)

    def reset(self):
        """Reset processor state"""
        self.signal_durations = []
        self.space_durations = []
        self.current_morse_pattern = []
        self.signal_buckets = StatisticalBuckets(n_buckets=2, std_devs=self.strictness)
        self.space_buckets = StatisticalBuckets(n_buckets=3, std_devs=self.strictness)


class ConfidenceDecoderGUI:
    def __init__(self, root):
        self.root = root

        # Live decoding state
        self.is_decoding = False
        self.audio_stream = None
        self.processing_lock = threading.Lock()

        # Multi-stage processors with individual settings
        self.processor_1s = DecodingProcessor("1s", 1.0, signal_lookback=20, space_lookback=20, strictness=0.05)
        self.processor_5s = DecodingProcessor("5s", 5.0, signal_lookback=20, space_lookback=20, strictness=0.05)
        self.processor_10s = DecodingProcessor("10s", 10.0, signal_lookback=20, space_lookback=20, strictness=0.05)
        self.processor_20s = DecodingProcessor("20s", 20.0, signal_lookback=20, space_lookback=20, strictness=0.05)

        # Separate audio buffers for each processor
        self.audio_buffer_1s = []
        self.audio_buffer_5s = []
        self.audio_buffer_10s = []
        self.audio_buffer_20s = []

        # Timing
        self.start_time = None
        self.last_1s_time = 0
        self.last_5s_time = 0
        self.last_10s_time = 0
        self.last_20s_time = 0

        # Individual controls for each processor
        self.strictness_1s = tk.DoubleVar(value=0.05)
        self.signal_lookback_1s = tk.IntVar(value=20)
        self.space_lookback_1s = tk.IntVar(value=20)

        self.strictness_5s = tk.DoubleVar(value=0.05)
        self.signal_lookback_5s = tk.IntVar(value=20)
        self.space_lookback_5s = tk.IntVar(value=20)

        self.strictness_10s = tk.DoubleVar(value=0.05)
        self.signal_lookback_10s = tk.IntVar(value=20)
        self.space_lookback_10s = tk.IntVar(value=20)

        self.strictness_20s = tk.DoubleVar(value=0.05)
        self.signal_lookback_20s = tk.IntVar(value=20)
        self.space_lookback_20s = tk.IntVar(value=20)

        # Confidence tracking - chunk-based approach
        self.text_chunks = []  # List of (text, time_start, time_end, confidence_level)
        self.text_lock = threading.Lock()

        root.title("Confidence-Based Statistical CW Decoder (1s/5s/10s/20s)")
        root.geometry("2000x1000")
        root.configure(bg='#2b2b2b')

        self.setup_ui()

    def setup_ui(self):
        # Title
        title_label = tk.Label(
            self.root,
            text="Confidence-Based CW Decoder (1s → 5s → 10s → 20s Progressive Refinement)",
            font=('Consolas', 16, 'bold'),
            bg='#2b2b2b',
            fg='#00ff00'
        )
        title_label.pack(pady=10)

        # Control panel
        control_frame = tk.Frame(self.root, bg='#1e1e1e', relief=tk.RIDGE, borderwidth=2)
        control_frame.pack(fill=tk.X, padx=20, pady=10)

        # Buttons
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

        # Individual processor controls in 4 columns
        processors_frame = tk.Frame(control_frame, bg='#1e1e1e')
        processors_frame.pack(fill=tk.X, padx=20, pady=10)

        self.create_processor_controls(processors_frame, "1s Processor", self.strictness_1s,
                                      self.signal_lookback_1s, self.space_lookback_1s,
                                      self.update_processor_1s, column=0)

        self.create_processor_controls(processors_frame, "5s Processor", self.strictness_5s,
                                      self.signal_lookback_5s, self.space_lookback_5s,
                                      self.update_processor_5s, column=1)

        self.create_processor_controls(processors_frame, "10s Processor", self.strictness_10s,
                                      self.signal_lookback_10s, self.space_lookback_10s,
                                      self.update_processor_10s, column=2)

        self.create_processor_controls(processors_frame, "20s Processor", self.strictness_20s,
                                      self.signal_lookback_20s, self.space_lookback_20s,
                                      self.update_processor_20s, column=3)

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

        # Main content container
        main_container = tk.Frame(self.root, bg='#2b2b2b')
        main_container.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)

        # Left side: Single confidence-based output
        output_container = tk.Frame(main_container, bg='#1e1e1e', relief=tk.RIDGE, borderwidth=2)
        output_container.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))

        tk.Label(
            output_container,
            text="Decoded Output (Progressive Confidence)",
            font=('Consolas', 12, 'bold'),
            bg='#1e1e1e',
            fg='#00ff00'
        ).pack(pady=5)

        # Legend
        legend_frame = tk.Frame(output_container, bg='#1e1e1e')
        legend_frame.pack(pady=5)

        tk.Label(legend_frame, text="● 1s: ", bg='#1e1e1e', fg='#505050', font=('Consolas', 8)).pack(side=tk.LEFT)
        tk.Label(legend_frame, text="Gray", bg='#1e1e1e', fg='#505050', font=('Consolas', 8, 'bold')).pack(side=tk.LEFT, padx=(0,10))

        tk.Label(legend_frame, text="● 5s: ", bg='#1e1e1e', fg='#888888', font=('Consolas', 8)).pack(side=tk.LEFT)
        tk.Label(legend_frame, text="White", bg='#1e1e1e', fg='#888888', font=('Consolas', 8, 'bold')).pack(side=tk.LEFT, padx=(0,10))

        tk.Label(legend_frame, text="● 10s: ", bg='#1e1e1e', fg='#00ff00', font=('Consolas', 8)).pack(side=tk.LEFT)
        tk.Label(legend_frame, text="Green", bg='#1e1e1e', fg='#00ff00', font=('Consolas', 8, 'bold')).pack(side=tk.LEFT, padx=(0,10))

        tk.Label(legend_frame, text="● 20s: ", bg='#1e1e1e', fg='#00ffff', font=('Consolas', 8)).pack(side=tk.LEFT)
        tk.Label(legend_frame, text="Cyan", bg='#1e1e1e', fg='#00ffff', font=('Consolas', 8, 'bold')).pack(side=tk.LEFT)

        self.output_text = tk.Text(
            output_container,
            height=20,
            bg='#2b2b2b',
            fg='#505050',
            font=('Consolas', 14),
            wrap=tk.WORD
        )

        # Configure tags for confidence levels
        self.output_text.tag_configure("conf_1s", foreground="#505050")  # Dim gray
        self.output_text.tag_configure("conf_5s", foreground="#888888")  # Medium white
        self.output_text.tag_configure("conf_10s", foreground="#00ff00")  # Bright green
        self.output_text.tag_configure("conf_20s", foreground="#00ffff")  # Cyan

        self.output_text.pack(padx=10, pady=(0, 10), fill=tk.BOTH, expand=True)

        # Right side: Console log
        console_container = tk.Frame(main_container, bg='#1e1e1e', relief=tk.RIDGE, borderwidth=2)
        console_container.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        tk.Label(
            console_container,
            text="Debug Console",
            font=('Consolas', 11, 'bold'),
            bg='#1e1e1e',
            fg='#FFC107'
        ).pack(pady=5)

        # Scrollbar for console
        console_scroll_frame = tk.Frame(console_container, bg='#1e1e1e')
        console_scroll_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))

        console_scrollbar = tk.Scrollbar(console_scroll_frame)
        console_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.console_text = tk.Text(
            console_scroll_frame,
            height=40,
            bg='#1a1a1a',
            fg='#00ff00',
            font=('Consolas', 8),
            wrap=tk.WORD,
            yscrollcommand=console_scrollbar.set
        )
        self.console_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        console_scrollbar.config(command=self.console_text.yview)

    def create_processor_controls(self, parent, title, strictness_var, signal_var, space_var, update_func, column):
        """Create individual controls for one processor"""
        frame = tk.Frame(parent, bg='#2b2b2b', relief=tk.RIDGE, borderwidth=1)
        frame.grid(row=0, column=column, padx=5, pady=5, sticky='nsew')
        parent.grid_columnconfigure(column, weight=1)

        # Title
        tk.Label(frame, text=title, bg='#2b2b2b', fg='#00ff00', font=('Consolas', 10, 'bold')).pack(pady=5)

        # Strictness
        tk.Label(frame, text="Strictness:", bg='#2b2b2b', fg='white', font=('Consolas', 8)).pack(anchor='w', padx=5)
        strictness_slider = tk.Scale(
            frame,
            from_=0.05,
            to=4.0,
            resolution=0.05,
            orient=tk.HORIZONTAL,
            variable=strictness_var,
            bg='#1e1e1e',
            fg='white',
            highlightthickness=0,
            length=150,
            command=update_func
        )
        strictness_slider.pack(padx=5, pady=2)

        self.__dict__[f'{title}_strict_label'] = tk.Label(frame, text=f"±{strictness_var.get():.2f}σ",
                                                          bg='#2b2b2b', fg='#00ff00', font=('Consolas', 8))
        self.__dict__[f'{title}_strict_label'].pack()

        # Signal lookback
        tk.Label(frame, text="Signal History:", bg='#2b2b2b', fg='white', font=('Consolas', 8)).pack(anchor='w', padx=5, pady=(5,0))
        signal_slider = tk.Scale(
            frame,
            from_=5,
            to=100,
            resolution=5,
            orient=tk.HORIZONTAL,
            variable=signal_var,
            bg='#1e1e1e',
            fg='white',
            highlightthickness=0,
            length=150,
            command=update_func
        )
        signal_slider.pack(padx=5, pady=2)

        self.__dict__[f'{title}_sig_label'] = tk.Label(frame, text=f"{signal_var.get()} signals",
                                                       bg='#2b2b2b', fg='#00ff00', font=('Consolas', 8))
        self.__dict__[f'{title}_sig_label'].pack()

        # Space lookback
        tk.Label(frame, text="Space History:", bg='#2b2b2b', fg='white', font=('Consolas', 8)).pack(anchor='w', padx=5, pady=(5,0))
        space_slider = tk.Scale(
            frame,
            from_=5,
            to=100,
            resolution=5,
            orient=tk.HORIZONTAL,
            variable=space_var,
            bg='#1e1e1e',
            fg='white',
            highlightthickness=0,
            length=150,
            command=update_func
        )
        space_slider.pack(padx=5, pady=2)

        self.__dict__[f'{title}_space_label'] = tk.Label(frame, text=f"{space_var.get()} spaces",
                                                         bg='#2b2b2b', fg='#00ff00', font=('Consolas', 8))
        self.__dict__[f'{title}_space_label'].pack()

    def update_processor_1s(self, value=None):
        """Update 1s processor settings"""
        strictness = self.strictness_1s.get()
        signal = self.signal_lookback_1s.get()
        space = self.space_lookback_1s.get()

        self.__dict__['1s Processor_strict_label'].config(text=f"±{strictness:.2f}σ")
        self.__dict__['1s Processor_sig_label'].config(text=f"{signal} signals")
        self.__dict__['1s Processor_space_label'].config(text=f"{space} spaces")

        self.processor_1s.update_strictness(strictness)
        self.processor_1s.update_lookback(signal, space)
        self.log(f"1s: strictness={strictness:.2f}, signal={signal}, space={space}")

    def update_processor_5s(self, value=None):
        """Update 5s processor settings"""
        strictness = self.strictness_5s.get()
        signal = self.signal_lookback_5s.get()
        space = self.space_lookback_5s.get()

        self.__dict__['5s Processor_strict_label'].config(text=f"±{strictness:.2f}σ")
        self.__dict__['5s Processor_sig_label'].config(text=f"{signal} signals")
        self.__dict__['5s Processor_space_label'].config(text=f"{space} spaces")

        self.processor_5s.update_strictness(strictness)
        self.processor_5s.update_lookback(signal, space)
        self.log(f"5s: strictness={strictness:.2f}, signal={signal}, space={space}")

    def update_processor_10s(self, value=None):
        """Update 10s processor settings"""
        strictness = self.strictness_10s.get()
        signal = self.signal_lookback_10s.get()
        space = self.space_lookback_10s.get()

        self.__dict__['10s Processor_strict_label'].config(text=f"±{strictness:.2f}σ")
        self.__dict__['10s Processor_sig_label'].config(text=f"{signal} signals")
        self.__dict__['10s Processor_space_label'].config(text=f"{space} spaces")

        self.processor_10s.update_strictness(strictness)
        self.processor_10s.update_lookback(signal, space)
        self.log(f"10s: strictness={strictness:.2f}, signal={signal}, space={space}")

    def update_processor_20s(self, value=None):
        """Update 20s processor settings"""
        strictness = self.strictness_20s.get()
        signal = self.signal_lookback_20s.get()
        space = self.space_lookback_20s.get()

        self.__dict__['20s Processor_strict_label'].config(text=f"±{strictness:.2f}σ")
        self.__dict__['20s Processor_sig_label'].config(text=f"{signal} signals")
        self.__dict__['20s Processor_space_label'].config(text=f"{space} spaces")

        self.processor_20s.update_strictness(strictness)
        self.processor_20s.update_lookback(signal, space)
        self.log(f"20s: strictness={strictness:.2f}, signal={signal}, space={space}")

    def log(self, message):
        """Add message to console log"""
        timestamp = time.strftime("%H:%M:%S")
        log_message = f"[{timestamp}] {message}\n"
        self.root.after(0, lambda: self._append_to_console(log_message))

    def _append_to_console(self, message):
        """Thread-safe append to console"""
        self.console_text.insert('end', message)
        self.console_text.see('end')

        # Keep console from growing too large (last 1000 lines)
        lines = int(self.console_text.index('end-1c').split('.')[0])
        if lines > 1000:
            self.console_text.delete('1.0', f'{lines-1000}.0')

    def toggle_decoding(self):
        """Toggle decoding on/off"""
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
        self.audio_buffer_1s = []
        self.audio_buffer_5s = []
        self.audio_buffer_10s = []
        self.audio_buffer_20s = []
        self.start_time = time.time()
        self.last_1s_time = 0
        self.last_5s_time = 0
        self.last_10s_time = 0
        self.last_20s_time = 0

        self.log("Decoding started - confidence-based output (1s/5s/10s/20s)")

        # Start audio stream
        try:
            self.audio_stream = sd.InputStream(
                samplerate=SAMPLE_RATE,
                channels=1,
                dtype='float32',
                blocksize=2000,
                callback=self.audio_callback
            )
            self.audio_stream.start()
        except Exception as e:
            self.log(f"Error starting audio stream: {e}")
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
        self.log("Decoding stopped")

    def audio_callback(self, indata, frames, time_info, status):
        """Callback for continuous audio streaming"""
        if status:
            self.log(f"Audio status: {status}")

        audio_chunk = indata[:, 0].copy()

        # Add to all buffers
        self.audio_buffer_1s.extend(audio_chunk)
        self.audio_buffer_5s.extend(audio_chunk)
        self.audio_buffer_10s.extend(audio_chunk)
        self.audio_buffer_20s.extend(audio_chunk)

        # Calculate elapsed time
        elapsed = time.time() - self.start_time

        # Trigger each processor independently at their intervals
        should_process_1s = elapsed - self.last_1s_time >= 1.0
        should_process_5s = elapsed - self.last_5s_time >= 5.0
        should_process_10s = elapsed - self.last_10s_time >= 10.0
        should_process_20s = elapsed - self.last_20s_time >= 20.0

        if should_process_1s:
            threading.Thread(target=self.process_1s, daemon=True).start()
            self.last_1s_time = elapsed

        if should_process_5s:
            threading.Thread(target=self.process_5s, daemon=True).start()
            self.last_5s_time = elapsed

        if should_process_10s:
            threading.Thread(target=self.process_10s, daemon=True).start()
            self.last_10s_time = elapsed

        if should_process_20s:
            threading.Thread(target=self.process_20s, daemon=True).start()
            self.last_20s_time = elapsed

    def process_audio_chunk(self, audio, processor):
        """Common audio processing pipeline"""
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
                return ""

            # Extract segments
            segments = extract_segments(binary_signal, frame_rate_hz=125)

            if not segments:
                return ""

            # Process with this processor
            decoded_text = processor.process_segments(segments)
            return decoded_text

        except Exception as e:
            self.log(f"Error in {processor.name} processor: {e}")
            import traceback
            traceback.print_exc()
            return ""

    def process_1s(self):
        """Process 1-second audio independently"""
        with self.processing_lock:
            if not self.is_decoding or len(self.audio_buffer_1s) < SAMPLE_RATE * 1.0:
                return

            audio = np.array(self.audio_buffer_1s)
            self.audio_buffer_1s = []

        current_time = time.time() - self.start_time
        self.log(f"=== 1s PROCESSOR at {current_time:.1f}s ===")

        decoded = self.process_audio_chunk(audio, self.processor_1s)

        if decoded:
            self.log(f"1s decoded: '{decoded}'")
            self.root.after(0, lambda d=decoded, t=current_time: self.update_confidence_text(d, confidence_level=1, timestamp=t, window_size=1.0))

    def process_5s(self):
        """Process 5-second audio independently"""
        with self.processing_lock:
            if not self.is_decoding or len(self.audio_buffer_5s) < SAMPLE_RATE * 5.0:
                return

            audio = np.array(self.audio_buffer_5s)
            self.audio_buffer_5s = []

        current_time = time.time() - self.start_time
        self.log(f"=== 5s PROCESSOR at {current_time:.1f}s ===")

        decoded = self.process_audio_chunk(audio, self.processor_5s)

        if decoded:
            self.log(f"5s decoded: '{decoded}'")
            self.root.after(0, lambda d=decoded, t=current_time: self.update_confidence_text(d, confidence_level=5, timestamp=t, window_size=5.0))

    def process_10s(self):
        """Process 10-second audio independently"""
        with self.processing_lock:
            if not self.is_decoding or len(self.audio_buffer_10s) < SAMPLE_RATE * 10.0:
                return

            audio = np.array(self.audio_buffer_10s)
            self.audio_buffer_10s = []

        current_time = time.time() - self.start_time
        self.log(f"=== 10s PROCESSOR at {current_time:.1f}s ===")

        decoded = self.process_audio_chunk(audio, self.processor_10s)

        if decoded:
            self.log(f"10s decoded: '{decoded}'")
            self.root.after(0, lambda d=decoded, t=current_time: self.update_confidence_text(d, confidence_level=10, timestamp=t, window_size=10.0))

    def process_20s(self):
        """Process 20-second audio independently"""
        with self.processing_lock:
            if not self.is_decoding or len(self.audio_buffer_20s) < SAMPLE_RATE * 20.0:
                return

            audio = np.array(self.audio_buffer_20s)
            self.audio_buffer_20s = []

        current_time = time.time() - self.start_time
        self.log(f"=== 20s PROCESSOR at {current_time:.1f}s ===")

        decoded = self.process_audio_chunk(audio, self.processor_20s)

        if decoded:
            self.log(f"20s decoded: '{decoded}'")
            self.root.after(0, lambda d=decoded, t=current_time: self.update_confidence_text(d, confidence_level=20, timestamp=t, window_size=20.0))

    def update_confidence_text(self, new_text, confidence_level, timestamp, window_size):
        """
        Update the output text with simple confidence-based replacement:
        - 1s: Always append gray text
        - 5s: Delete all gray (conf=1), append white text
        - 10s: Delete all white (conf=5), append green text
        - 20s: Delete all green (conf=10), append cyan text
        """
        if not new_text:
            return

        with self.text_lock:
            self.log(f"Updating with {confidence_level}s output: '{new_text}'")

            # Determine which confidence level to remove
            # 5s removes all 1s, 10s removes all 5s, 20s removes all 10s
            conf_to_remove = None
            if confidence_level == 5:
                conf_to_remove = 1
            elif confidence_level == 10:
                conf_to_remove = 5
            elif confidence_level == 20:
                conf_to_remove = 10

            # If we should remove a lower confidence level, do it
            if conf_to_remove is not None:
                chunks_to_remove = []
                char_position = 0

                for idx, (chunk_text, chunk_start, chunk_end, chunk_conf) in enumerate(self.text_chunks):
                    if chunk_conf == conf_to_remove:
                        chunks_to_remove.append((idx, char_position, len(chunk_text)))
                        self.log(f"  Removing conf={conf_to_remove} chunk: '{chunk_text}'")
                    char_position += len(chunk_text)

                # Remove chunks from text widget and list (in reverse order)
                for idx, char_pos, length in reversed(chunks_to_remove):
                    start_pos = f"1.0 + {char_pos} chars"
                    end_pos = f"1.0 + {char_pos + length} chars"
                    self.output_text.delete(start_pos, end_pos)
                    del self.text_chunks[idx]

                self.log(f"  Removed {len(chunks_to_remove)} chunks with conf={conf_to_remove}")

            # Append new text at the end
            insert_char_pos = sum(len(chunk[0]) for chunk in self.text_chunks)
            time_start = timestamp - window_size
            time_end = timestamp

            self.text_chunks.append((new_text, time_start, time_end, confidence_level))

            # Insert into text widget
            insert_pos = f"1.0 + {insert_char_pos} chars"
            self.output_text.insert(insert_pos, new_text)

            # Apply confidence tag
            tag_name = f"conf_{confidence_level}s"
            start_idx = f"1.0 + {insert_char_pos} chars"
            end_idx = f"1.0 + {insert_char_pos + len(new_text)} chars"
            self.output_text.tag_add(tag_name, start_idx, end_idx)

            self.output_text.see('end')
            total_chars = sum(len(chunk[0]) for chunk in self.text_chunks)
            self.status_label.config(text=f"Status: {confidence_level}s decoded - {total_chars} chars total")
            self.log(f"  Total chunks: {len(self.text_chunks)}, total chars: {total_chars}")

    def clear_data(self):
        """Clear all data and reset"""
        with self.processing_lock:
            self.audio_buffer_1s = []
            self.audio_buffer_5s = []
            self.audio_buffer_10s = []
            self.audio_buffer_20s = []
            self.processor_1s.reset()
            self.processor_5s.reset()
            self.processor_10s.reset()
            self.processor_20s.reset()

        with self.text_lock:
            self.output_text.delete('1.0', tk.END)
            self.text_chunks = []

        self.status_label.config(text="Status: Data cleared - ready for new signal")
        self.log("Data cleared - all processors reset")


if __name__ == '__main__':
    root = tk.Tk()
    app = ConfidenceDecoderGUI(root)
    root.mainloop()

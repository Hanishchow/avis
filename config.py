"""Central configuration for the bird detection system."""

from __future__ import annotations

# ── Audio ─────────────────────────────────────────────
SAMPLE_RATE = 44100
CHANNELS = 2
WINDOW_DURATION = 15  # seconds
DEVICE_INDEX = 1

# ── Feature extraction (mel-spectrogram) ───────────────
N_MELS = 128
N_FFT = 2048
HOP_LENGTH = 512
FMIN = 500
FMAX = 16000
DURATION = 3.0  # seconds per clip
N_SAMPLES = int(SAMPLE_RATE * DURATION)  # 132300

# Expected spectrogram shape
SPEC_H = N_MELS          # 128
SPEC_W = 259  # derived from SR * DUR / HOP_LENGTH + 1

# ── Filtering pipeline ────────────────────────────────
HP_CUTOFF = 300
LP_CUTOFF = 22000
BAND_LOWCUT = 500
FILTER_ORDER = 5
NOTCH_FREQS = [4500, 5500, 6500, 7500, 8500, 9500]
NOTCH_Q = 30
WIENER_PROP_DECREASE = 0.8
NOISE_FLOOR_DB = -50
SPECTRAL_GATE_NPERSEG = 1024
SPECTRAL_GATE_NOVERLAP = 768
NORMALIZE_TARGET_MAX = 32767
STATIONARY_WIENER = True
ENABLE_WIENER = True
ENABLE_NOTCH = True

# ── Ensemble / detection ──────────────────────────────
DETECTION_THRESHOLD = 0.35
BIRDNET_WEIGHT = 0.7
NN_WEIGHT = 0.3
FALLBACK_COMBINE_THRESHOLD = DETECTION_THRESHOLD

# ── BirdNET integration ───────────────────────────────
LAT = 12.97
LON = 77.59
BIRDNET_MIN_CONF = 0.35
TOP_K_LOGITS = 3

# ── VAD ────────────────────────────────────────────────
SPEECH_PROB_THRESHOLD = 0.4

# ── Directories / file paths ──────────────────────────
DATASET_DIR = "real_dataset"
AUDIO_DIR = f"{DATASET_DIR}/audio"
SEG_DIR = f"{DATASET_DIR}/segments"
SPEC_DIR = f"{DATASET_DIR}/spectrograms"
METADATA_DIR = f"{DATASET_DIR}/metadata"
OUTPUT_DIR = "detections"
LOG_FILE = "birdnet_enhanced.log"

# Model paths
CNN_MODEL_PATH = "bird_sound_classifier.h5"
MLP_MODEL_PATH = "mlp_stacker.h5"
MLP_TRAINING_DATA = f"{DATASET_DIR}/mlp_training_data.csv"

# ── Xeno-Canto API ────────────────────────────────────
XENO_CANTO_API_KEY = "dc05237b5fb65a8d3f6c19665aa4c0a2fa39a6f7"
XENO_CANTO_PAGE_SIZE = "40"
DOWNLOAD_TIMEOUT = 45
DOWNLOAD_CHUNK_SIZE = 8192
DOWNLOAD_SLEEP = 1.0  # seconds between requests

# ── Training ──────────────────────────────────────────
CNN_EPOCHS = 50
CNN_BATCH_SIZE = 8
CNN_EARLY_STOP_PATIENCE = 5
CNN_LR_FACTOR = 0.5
CNN_LR_PATIENCE = 3
CNN_MIN_LR = 1e-6
CNN_RANDOM_STATE = 42
CNN_TEST_SPLIT = 0.2

MLP_EPOCHS = 50
MLP_BATCH_SIZE = 16
MLP_HIDDEN_DIM = 8
MLP_L2_REG = 0.0
MLP_DROPOUT = 0.2
MLP_USE_CLASS_WEIGHT = True
MLP_EARLY_STOP_PATIENCE = 5
MLP_RANDOM_STATE = 42
MLP_TEST_SPLIT = 0.2

# ── Evaluation ────────────────────────────────────────
CV_FOLDS = 5
FP_PER_HOUR_DIVISOR = 240.0
EVAL_THRESHOLDS = [0.35, 0.40, 0.50, 0.60, 0.70, 0.80]
SWEEP_THRESHOLDS = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.80]

# Targets
SC007_TARGET_F1 = 0.75
SC008_TARGET_FP_HR = 0.5

# ── Synthetic noise ──────────────────────────────────
NUM_SYNTHETIC_NOISE = 80
SYNTHETIC_NOISE_TYPES = ["white", "pink", "brown", "wind", "crackle"]
NOISE_AMPLITUDE = 0.3
NOISE_NORMALIZE_CEILING = 0.3
NOISE_PREFIXES = {"rain", "wind", "stream"}
NOISE_PREFIXES_TUPLE = ("rain_", "wind_", "stream_", "noise_")

# ── Dataset preparation ──────────────────────────────
DATASET_TEST_SPLIT = 0.15
DATASET_VAL_SPLIT = 0.1765
DATASET_RANDOM_STATE = 42
LOAD_MAX_DURATION = 60

# ── Sweep configs ─────────────────────────────────────
MLP_SWEEP_CONFIGS = [
    (8, 0.0,   0.2, "baseline   (hd=8,  l2=0,   drop=0.2)"),
    (8, 1e-4,  0.2, "l2-1e4     (hd=8,  l2=1e4, drop=0.2)"),
    (16, 1e-4,  0.3, "l2-1e4-hd16(hd=16, l2=1e4, drop=0.3)"),
    (32, 1e-4,  0.3, "l2-1e4-hd32(hd=32, l2=1e4, drop=0.3)"),
    (16, 1e-3,  0.3, "l2-1e3-hd16(hd=16, l2=1e3, drop=0.3)"),
    (32, 1e-3,  0.3, "l2-1e3-hd32(hd=32, l2=1e3, drop=0.3)"),
]

EVAL_SWEEP_CONFIGS = [
    (8,  0.0,   0.2, "baseline"),
    (8,  1e-4,  0.2, "l2-1e4"),
    (16, 1e-4,  0.3, "hd16-l2e4"),
    (32, 1e-4,  0.3, "hd32-l2e4"),
    (16, 1e-3,  0.3, "hd16-l2e3"),
    (32, 1e-3,  0.3, "hd32-l2e3"),
    (8,  1e-4,  0.3, "l2-bigdrop"),
]

# ── Xeno-Canto species queries ────────────────────────
NOISE_QUERIES = [
    ("en:rain q:A", "rain"),
    ("en:wind q:A", "wind"),
    ("en:stream q:A", "stream"),
]

BIRD_QUERIES = [
    ("gen:Columba sp:livia", "Rock_Pigeon"),
    ("gen:Copsychus sp:saularis", "Oriental_Magpie-Robin"),
    ("gen:Prinia sp:socialis", "Ashy_Prinia"),
    ("gen:Dicrurus sp:macrocercus", "Black_Drongo"),
    ("gen:Acridotheres sp:tristis", "Common_Myna"),
    ("gen:Passer sp:domesticus", "House_Sparrow"),
    ("gen:Centropus sp:sinensis", "Greater_Coucal"),
    ("gen:Orthotomus sp:sutorius", "Common_Tailorbird"),
    ("gen:Cinnyris sp:asiaticus", "Purple_Sunbird"),
    ("gen:Halcyon sp:smyrnensis", "White-breasted_Kingfisher"),
]

DOWNLOAD_SPECIES = [
    ("Acridotheres", "tristis", "Common_Myna"),
    ("Passer", "domesticus", "House_Sparrow"),
    ("Corvus", "splendens", "House_Crow"),
    ("Pycnonotus", "cafer", "Red-vented_Bulbul"),
    ("Columba", "livia", "Rock_Dove"),
    ("Copsychus", "saularis", "Oriental_Magpie-Robin"),
    ("Prinia", "socialis", "Ashy_Prinia"),
    ("Dicrurus", "macrocercus", "Black_Drongo"),
    ("Centropus", "sinensis", "Greater_Coucal"),
    ("Orthotomus", "sutorius", "Common_Tailorbird"),
    ("Cinnyris", "asiaticus", "Purple_Sunbird"),
    ("Halcyon", "smyrnensis", "White-breasted_Kingfisher"),
]

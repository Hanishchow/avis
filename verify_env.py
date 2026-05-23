"""Verify that all required dependencies are installed and log their versions."""
import sys
import importlib

DEPS = [
    ("pyaudio", "pyaudio"),
    ("numpy", "numpy"),
    ("scipy", "scipy"),
    ("librosa", "librosa"),
    ("noisereduce", "noisereduce"),
    ("tensorflow", "tensorflow"),
    ("matplotlib", "matplotlib"),
    ("birdnetlib", "birdnetlib"),
    ("silero_vad", "silero_vad"),
    ("scikit-learn", "sklearn"),
    ("pandas", "pandas"),
    ("seaborn", "seaborn"),
]

print(f"Python version: {sys.version}")
ok = True
for name, pkg in DEPS:
    try:
        m = importlib.import_module(pkg)
        ver = getattr(m, "__version__", "unknown")
        print(f"  {name}: {ver}")
    except ImportError as e:
        print(f"  {name}: MISSING - {e}")
        ok = False

if ok:
    print("\nAll dependencies import successfully!")
    sys.exit(0)
else:
    print("\nSome dependencies are missing!")
    sys.exit(1)

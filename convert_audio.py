import librosa
import soundfile as sf
audio, sr = librosa.load("test_bird_Delicate_Prinia.mp3", sr=44100, mono=True)
sf.write("test_bird_real.wav", audio, sr)
print(f"Converted: {len(audio)} samples, {len(audio)/sr:.1f}s, {sr}Hz")

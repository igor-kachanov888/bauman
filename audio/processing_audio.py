import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt

audio_path = r'/home/kali/Desktop/audio-processing/T08-violin.wav'

x, sr = librosa.load(audio_path)
print(type(x), type(sr))
print(x.shape, sr)

librosa.display.specshow

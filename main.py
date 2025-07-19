#https://github.com/Camb-ai/MARS5-TTS?tab=readme-ov-file

import torch
import librosa
import sounddevice as sd
from scipy.io.wavfile import write
wave_sample = 'Untitled.wav'

ref_transcript ="Anyone who is condesending and judgemental about the right way to do something is also"
# Load the Mars5 TTS model and configuration class
mars5, config_class = torch.hub.load('Camb-ai/mars5-tts', 'mars5_english', trust_repo=True)

# The `mars5` contains the AR and NAR model, as well as inference code.
# The `config_class` contains tunable inference config settings like temperature.

# Load reference audio between 1-12 seconds.
wav, sr = librosa.load( wave_sample, sr=mars5.sr, mono=False)
wav = torch.from_numpy(wav)

# Pick whether you want a deep or shallow clone. Set to False if you don't know prompt transcript or want fast inference. Set to True if you know transcript and want highest quality.
deep_clone = True
# Below you can tune other inference settings, like top_k, temperature, top_p, etc...
cfg = config_class(deep_clone=deep_clone, rep_penalty_window=100, top_k=100, temperature=0.7, freq_penalty=3)
ar_codes, output_audio = mars5.tts("Hello and welcome.  I hope you can hear me clearly.", wav, ref_transcript, cfg=cfg)
# Convert the output to a numpy array and save it as a .wav file
output_audio_np = output_audio.cpu().numpy()
write('output_audio.wav', mars5.sr, output_audio_np)

# Optionally, play the audio
sd.play(output_audio_np, mars5.sr)
sd.wait()
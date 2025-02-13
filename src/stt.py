import logging
import numpy as np
import sounddevice as sd
from librosa import resample
from faster_whisper import WhisperModel
from time import time

class STT:
    def __init__(self, mic_name, model_size: str = "base", device: str = "cpu", compute_type: str = "int8"):
        self.model = WhisperModel(model_size, device=device, compute_type=compute_type)
        self.logger = logging.getLogger("STT")

        self.device_idx = None
        for i, device in enumerate(sd.query_devices()):
            if device['max_input_channels'] > 0 and mic_name.lower() in device['name'].lower():
                self.logger.debug(f"Found device {mic_name} at index {i}")
                self.device_idx = i
                break
        
        if self.device_idx is None:
            raise RuntimeError(f"Could not find device {mic_name}")
        
    def record_transcribe(self, duration=6, samplerate=44100, channels=2):
        print(f"Recording!")
        start = time()
        audio_data = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=channels, device=self.device_idx, dtype=np.float32)
        sd.wait()

        resampled_audio = resample(audio_data[:,0].flatten(), orig_sr=samplerate, target_sr=16000)

        print(f"Transcribing")
        segments, _ = self.model.transcribe(resampled_audio, language="en", beam_size=5)

        print(f"Printing?!")
        for segment in segments:
            print(f"{segment.start:.2f}s - {segment.end:.2f}s: {segment.text}")
        
        end = time()

        print(f"Total Processing Time: {end - start - duration}")
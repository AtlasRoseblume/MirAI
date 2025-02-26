import logging
import numpy as np
import sounddevice as sd
from librosa import resample

class Microphone:
    def __init__(self, mic_name:str):
        self.logger = logging.getLogger("Microphone")
        self.device_idx = None
        self.audio_buffer = None
        for i, device in enumerate(sd.query_devices()):
        
            if device['max_input_channels'] > 0 and mic_name.lower() in device['name'].lower():
                self.logger.debug(f"Found device {mic_name} at index {i}")
                self.device_idx = i
                break

        if self.device_idx is None:
            raise RuntimeError(f"Could not find microphone with name: {mic_name}")

    def record(self, duration=5, samplerate=44100, target_sr=16000, channels=2) -> np.ndarray:
        if self.audio_buffer is None:
            self.audio_buffer = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=channels, device=self.device_idx, dtype=np.float32)
        
        sd.wait()

        new_audio_buffer = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=channels, device=self.device_idx, dtype=np.float32)

        self.logger.debug(f"Recorded {duration} seconds")
        resampled = resample(self.audio_buffer[:,0].flatten(), orig_sr=samplerate, target_sr=target_sr)

        self.audio_buffer = new_audio_buffer
        return resampled
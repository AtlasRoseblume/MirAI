import numpy as np
import sounddevice as sd
from piper.voice import PiperVoice

class TTS:
    def __init__(self, model_path: str):
        self.voice = PiperVoice.load(model_path)
        self.stream = sd.OutputStream(samplerate=self.voice.config.sample_rate, channels=1, dtype='int16')
        self.stream.start()
    
    def say(self, text: str, speed: float = 0.8):
        # We want to synthesize at max speed, so no silence
        for audio_bytes in self.voice.synthesize_stream_raw(text, length_scale=speed, sentence_silence=0.0):
            self.stream.write(np.frombuffer(audio_bytes, dtype=np.int16))

    def close(self):
        self.stream.stop()
        self.stream.close()
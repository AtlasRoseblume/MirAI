import logging
import numpy as np
import sounddevice as sd
from piper.voice import PiperVoice

class TTS:
    def __init__(self, model_path: str):
        self.logger = logging.getLogger("TTS")
        self.voice = PiperVoice.load(model_path)
        self.stream = sd.OutputStream(samplerate=self.voice.config.sample_rate, channels=1, dtype='int16')
        self.stream.start()
        self.logger.info(f"Voice Model Loaded from {model_path}")
    
    def say(self, text: str):
        self.logger.debug(f"Printing \"{text}\"")

        # We want to synthesize at max speed, so no silence
        for audio_bytes in self.voice.synthesize_stream_raw(text):
            self.stream.write(np.frombuffer(audio_bytes, dtype=np.int16))

    def close(self):
        self.stream.stop()
        self.stream.close()
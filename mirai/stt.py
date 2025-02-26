import logging
from faster_whisper import WhisperModel
from time import time

class STT:
    def __init__(self, model_size: str = "base.en", device: str = "cpu", compute_type: str = "int8"):
        self.model = WhisperModel(model_size, device=device, compute_type=compute_type)
        self.logger = logging.getLogger("STT")
        
    def transcribe(self, audio_data) -> str:
        start = time()

        segments, _ = self.model.transcribe(audio_data, language="en", beam_size=5, vad_filter=True)

        text = []
        for segment in segments:
            text.append(segment.text)
        
        end = time()
        self.logger.debug(f"Total Processing Time: {end - start}")

        return "".join(text).lstrip()
import logging
import os
import numpy as np
from datetime import datetime
from threading import Thread
from queue import Queue
from soundfile import SoundFile
from argparse import ArgumentParser
from .mic import Microphone
from .stt import STT
from .tts import TTS

class MirAI:
    def __init__(self, voice_path: str, microphone: str, record_mode: bool = False):
        self.logger =logging.getLogger("MirAI")
        self.running = True

        self.microphone = Microphone(microphone)
        self.tts = TTS(voice_path)
        self.stt = STT()

        self.audio_queue = Queue(maxsize=2)
        self.listener_thread = Thread(target=MirAI.audio_listener, args=(self,))
        self.listener_thread.start()

        self.listening = True
        self.recording = record_mode

    def audio_listener(self):
        while self.running:
            audio_clip = self.microphone.record(5)
            self.audio_queue.put_nowait(audio_clip) # Could error (but how?)

    def run(self):
        if self.recording:
            os.makedirs('./out/recordings', exist_ok=True)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_filename = os.path.join('./out/recordings', f'recording_{timestamp}.flac')
            file = SoundFile(output_filename, mode='w', samplerate=16000, channels=1, format='FLAC')

        while True:
            try:
                audio_clip = self.audio_queue.get()

                if self.recording:
                    file.write(audio_clip)
                    file.flush()

                if self.listening:
                    text = self.stt.transcribe(audio_clip)
                    print(text)
                else:
                    self.logger.debug("Discarded, not listening currently!")

            except KeyboardInterrupt:
                self.running = False
                print("Exiting normally.")
                break
        
        if self.recording:
            file.close()


def main():
    logging.basicConfig(filename="stt.log", level=logging.DEBUG)
    logging.getLogger("faster_whisper").setLevel(logging.DEBUG)

    parser = ArgumentParser(description="Start Robot Waifu Program")
    parser.add_argument('-r', '--record', action='store_true', help="Record microphone and video")
    parser.add_argument('-m', '--microphone', required=True, type=str, help="Micrpohone Name")
    parser.add_argument('-v', '--voice', required=True, type=str, help="Voice Model File (.onnx)")

    args = parser.parse_args()

    mirai = MirAI(args.voice, args.microphone, args.record)
    mirai.run()
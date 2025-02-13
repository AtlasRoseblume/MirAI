import logging
from threading import Thread
from queue import Queue
from time import time
from argparse import ArgumentParser
from .mic import Microphone
from .stt import STT
from .tts import TTS

class MirAI:
    def __init__(self, voice_path: str, microphone: str):
        self.logger =logging.getLogger("MirAI")
        self.running = True

        self.microphone = Microphone(microphone)
        self.tts = TTS(voice_path)
        self.stt = STT()

        self.audio_queue = Queue(maxsize=2)
        self.listener_thread = Thread(target=MirAI.audio_listener, args=(self,))
        self.listener_thread.start()

        self.listening = True

    def audio_listener(self):
        while self.running:
            audio_clip = self.microphone.record(5)
            self.audio_queue.put_nowait(audio_clip) # Could error (but how?)

    def run(self):
        while True:
            try:
                audio_clip = self.audio_queue.get()

                if self.listening:
                    text = self.stt.transcribe(audio_clip)
                    print(text)
                else:
                    self.logger.debug("Discarded, not listening currently!")

            except KeyboardInterrupt:
                self.running = False
                print("Exiting normally.")
                break


def main():
    logging.basicConfig(filename="stt.log", level=logging.DEBUG)
    logging.getLogger("faster_whisper").setLevel(logging.DEBUG)

    parser = ArgumentParser(description="Start Robot Waifu Program")
    parser.add_argument('-m', '--microphone', required=True, type=str, help="Micrpohone Name")
    parser.add_argument('-v', '--voice', required=True, type=str, help="Voice Model File (.onnx)")

    args = parser.parse_args()

    mirai = MirAI(args.voice, args.microphone)
    mirai.run()
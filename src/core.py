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

        self.audio_queue = Queue(maxsize=12) # Set this to an implausibly high number (60 seconds backlog)
        self.listener_thread = Thread(target=MirAI.audio_listener, args=(self,))
        self.listener_thread.start()

        self.listening = True
        self.recording = record_mode

        self.buffer = ""
        self.start_index = None

        self.wake_strings = ["hello world"]
        self.end_strings = ["goodbye world"]

    def audio_listener(self):
        while self.running:
            audio_clip = self.microphone.record(5)
            self.audio_queue.put_nowait(audio_clip) # Could error (but how?)

    @staticmethod
    def find_last_occurence(string, substrings):
        return max((string.lower().rfind(sub.lower()) for sub in substrings), default=-1)

    def find_trigger(self, text: str):
        self.buffer += text
        self.buffer += " "
        print(self.buffer)

        if self.start_index is None:
            start_match = MirAI.find_last_occurence(self.buffer, self.wake_strings)
            
            if start_match != -1:
                self.start_index = start_match + len('hello world')
        
        if self.start_index is not None:
            end_index = MirAI.find_last_occurence(self.buffer, self.end_strings)

            if end_index != -1:
                capture = self.buffer[self.start_index:end_index].lstrip("., \n\t").capitalize()
                print(f"Phrase submit {capture}")

                # TODO: Submit to SLM/LLM Queue
                # TODO: Disable listening, wait for response finalization

                # Reset State
                self.buffer = ""
                self.start_index = None


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
                    self.find_trigger(text)
                else:
                    self.logger.debug("Discarded, not listening currently!")

            except KeyboardInterrupt:
                self.running = False
                print("Exiting normally.")

                self.listener_thread.join()
                print("Listener joined.")
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
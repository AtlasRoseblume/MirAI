import logging
import json
import os
from concurrent.futures import TimeoutError, ThreadPoolExecutor
from datetime import datetime
from threading import Thread
from queue import Queue
from soundfile import SoundFile
from time import time
from argparse import ArgumentParser
from .mic import Microphone
from .model import Model
from .stt import STT
from .ui import UI

class MirAI:
    def __init__(self, voice_path: str, microphone: str, llm_path: str, images_path: str, record_mode: bool = False, headless_mode: bool = False, config_path: str = 'config.json'):
        self.logger =logging.getLogger("MirAI")
        self.running = True

        self.microphone = Microphone(microphone)
        self.stt = STT()

        self.audio_queue = Queue(maxsize=12) # Set this to an implausibly high number (60 seconds backlog)
        self.listener_thread = Thread(target=MirAI.audio_listener, args=(self,))
        
        self.listening = True
        self.recording = record_mode
        self.headless = headless_mode

        self.buffer = ""
        self.captured_text = ""
        self.response_buffer = ""
        self.start_index = None

        self.listen_time = time()

        default_wake_phrases = ["hello world"]
        default_end_phrases = ["goodbye world"]

        base_host = "127.0.0.1"
        base_port = 8000

        cheat_host = "1.1.1.1"
        cheat_port = 8000

        try:
            with open(config_path, 'r') as file:
                data = json.load(file)
            
            self.wake_strings = data.get('wake_phrases', default_wake_phrases)
            self.end_strings = data.get('end_phrases', default_end_phrases)

            base_host = data["base_host"]
            base_port = data["base_port"]

            cheat_host = data["cheat_host"]
            cheat_port = data["cheat_port"]
        except (FileNotFoundError, json.JSONDecodeError) as e:
            self.logger.error(f"Failed to load JSON File: {e}")
            self.wake_strings = default_wake_phrases
            self.end_strings = default_end_phrases

        self.model = Model(llm_path, voice_path, self, cheat_host=cheat_host, cheat_port=cheat_port, host=base_host, port=base_port)

        if not self.headless:
            self.ui = UI(self, images_path)

        # Start listen thread at the end so it's not offset
        self.listener_thread.start()


    def audio_listener(self):
        while self.running:
            audio_clip = self.microphone.record(5)
            self.audio_queue.put_nowait(audio_clip) # Could error (but how?)

    @staticmethod
    def find_first_occurence(string, substrings):
        earliest_index = float('inf')
        found_substring = None

        for sub in substrings:
            index = string.lower().find(sub.lower())

            if 0 <= index < earliest_index:
                earliest_index = index
                found_substring = sub
        
        if found_substring is not None:
            return earliest_index, len(found_substring)
        return -1, 0

    @staticmethod
    def find_last_occurence(string, substrings):
        last_index = -1
        
        for sub in substrings:
            index = string.lower().rfind(sub.lower())

            if index > last_index:
                last_index = index
        
        return last_index

    def find_trigger(self, text: str):
        self.buffer += text

        if len(text) != 0:
            self.buffer += " "
        
        print(f"History: {self.buffer}")

        if self.start_index is None:
            start_match, strlen = MirAI.find_first_occurence(self.buffer, self.wake_strings)
            
            if start_match != -1:
                self.start_index = start_match + strlen
        
        if self.start_index is not None:
            end_index = MirAI.find_last_occurence(self.buffer, self.end_strings)

            if end_index != -1:
                # Occurs when partial sentence fragment fails
                if end_index < self.start_index:
                    return

                capture = self.buffer[self.start_index:end_index].lstrip("., \n\t").capitalize()
                print(f"You said: {capture}")
                self.captured_text = capture
                self.response_buffer = ""

                self.listening = False
                self.model.queue.put_nowait((capture, self))

                # Reset State
                self.buffer = ""
                self.start_index = None


    def run(self):
        print("MirAI ready!")

        if self.recording:
            os.makedirs('./out/recordings', exist_ok=True)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_filename = os.path.join('./out/recordings', f'recording_{timestamp}.flac')
            file = SoundFile(output_filename, mode='w', samplerate=16000, channels=1, format='FLAC')

        while self.running:
            try:
                audio_clip = self.audio_queue.get()

                if self.recording:
                    file.write(audio_clip)
                    file.flush()

                current_time = time()
                if current_time - self.listen_time < 5:
                    continue

                if self.listening:
                    with ThreadPoolExecutor() as executor:
                        future = executor.submit(lambda: self.stt.transcribe(audio_clip))

                        try:
                            text = future.result(timeout=10)
                            self.find_trigger(text)
                        except TimeoutError:
                            self.logger.info("Whisper Timed Out!")

                if not self.headless:
                    self.running = self.ui.running

            except KeyboardInterrupt:
                self.running = False

                if not self.headless:
                    self.ui.running = False
                
                break

        print("Exiting normally.")

        self.listener_thread.join()
        print("Listener joined.")

        self.model.close()
        print("Closed llama-server")
        
        if self.recording:
            file.close()
    
    def toggle_cheat_mode(self):
        self.logger.info(f"Cheat Mode Set: {not self.model.cheat_mode}")
        self.model.cheat_mode = not self.model.cheat_mode


def main():
    logging.basicConfig(filename="mirai.log", level=logging.DEBUG)
    logging.getLogger("faster_whisper").setLevel(logging.DEBUG)

    parser = ArgumentParser(description="Start Robot Waifu Program")
    parser.add_argument('-c', '--config', type=str, default="config.json", help="Default configuration file")
    parser.add_argument('-n', '--no_window', action='store_true', help="Run Headless Mode")
    parser.add_argument('-i', '--images', type=str, default="images", help="Path to images directory for display")
    parser.add_argument('-l', '--llm', required=True, type=str, help="Model path (.gguf)")
    parser.add_argument('-m', '--microphone', required=True, type=str, help="Micrpohone Name")
    parser.add_argument('-r', '--record', action='store_true', help="Record microphone and video")
    parser.add_argument('-v', '--voice', required=True, type=str, help="Voice Model File (.onnx)")

    args = parser.parse_args()

    mirai = MirAI(args.voice, args.microphone, args.llm, args.images, args.record, args.no_window)
    mirai.run()
import base64
import logging
import io
from PIL import Image
import json
import multiprocessing
import multiprocessing.process
import os
import cv2
from concurrent.futures import TimeoutError, ThreadPoolExecutor
from datetime import datetime
from threading import Thread
from queue import Queue
from soundfile import SoundFile
from time import time, sleep
from argparse import ArgumentParser
from .mic import Microphone
from .model import Model
from .stt import STT
from .ui import UI

class MirAI:
    def __init__(self, manager, voice_path: str, microphone: str, llm_path: str, images_path: str, record_mode: bool = False, headless_mode: bool = False, config_path: str = 'config.json'):
        self.logger =logging.getLogger("MirAI")

        self.audio_queue = multiprocessing.Queue(maxsize=12) # Set this to an implausibly high number (60 seconds backlog)
        self.transcription_queue = multiprocessing.Queue()
        
        self.shared_state = manager.dict()
        self.shared_state["running"] = True
        self.shared_state["listening"] = True

        self.listener_thread = multiprocessing.Process(target=MirAI.audio_listener, args=(self, microphone, self.shared_state))
        self.transcription_thread = multiprocessing.Process(target=MirAI.run_transcription, args=(self, self.shared_state))
        
        self.recording = record_mode
        self.headless = headless_mode

        self.buffer = ""
        self.captured_text = ""
        self.response_buffer = ""
        self.start_index = None

        self.listen_time = time()

        default_wake_phrases = ["hello world"]
        default_end_phrases = ["goodbye world"]
        default_picture_phrases = ["take a picture"]

        base_host = "127.0.0.1"
        base_port = 8000

        cheat_host = "1.1.1.1"
        cheat_port = 8000

        prompt = "You are a useful AI assistant. Please help the user."

        try:
            with open(config_path, 'r') as file:
                data = json.load(file)
            
            self.wake_strings = data.get('wake_phrases', default_wake_phrases)
            self.end_strings = data.get('end_phrases', default_end_phrases)
            self.picture_strings = data.get('picture_phrases', default_picture_phrases)

            base_host = data["base_host"]
            base_port = data["base_port"]

            cheat_host = data["cheat_host"]
            cheat_port = data["cheat_port"]

            prompt = data["prompt"]
        except (FileNotFoundError, json.JSONDecodeError) as e:
            self.logger.error(f"Failed to load JSON File: {e}")
            self.wake_strings = default_wake_phrases
            self.end_strings = default_end_phrases
            self.picture_strings = default_picture_phrases

        self.model = Model(llm_path, voice_path, self, prompt=prompt, cheat_host=cheat_host, cheat_port=cheat_port, host=base_host, port=base_port)

        # Start listen thread at the end so it's not offset
        self.listener_thread.start()
        self.transcription_thread.start()

        if not self.headless:
            self.ui = UI(self)


    def audio_listener(self, mic_name: str, shared_state):
        if self.recording:
            os.makedirs('./out/recordings', exist_ok=True)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_filename = os.path.join('./out/recordings', f'recording_{timestamp}.flac')
            file = SoundFile(output_filename, mode='w', samplerate=16000, channels=1, format='FLAC')
        
        microphone = Microphone(mic_name)

        while shared_state["running"]:
            try:
                audio_clip = microphone.record(5)
                
                if self.recording:
                    file.write(audio_clip)
                    file.flush()
                
                self.audio_queue.put_nowait(audio_clip) # Could error (but how?)
            except KeyboardInterrupt:
                break

        if self.recording:
            file.close() 

        shared_state["listening"] = False
        self.audio_queue.put_nowait([])

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

                picture_taken = False
                picture = None
                for sub in self.picture_strings:
                    if sub.lower() in capture.lower():

                        try:
                            cap = cv2.VideoCapture(0)

                            if not cap.isOpened():
                                self.model.tts.say("Could not open camera!")
                                break
                        
                            cv2.waitKey(1000)
                            self.model.tts.say("Taking picture in 3, 2, 1!")

                            ret, frame = cap.read()

                            if not ret:
                                self.model.tts.say("Could not capture frame!")
                                break

                            image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                            buffer = io.BytesIO()
                            image.save(buffer, format = "PNG")

                            picture = base64.b64encode(buffer.getvalue()).decode('utf-8')
                            picture_taken = True
                        except Exception as e:
                            print(f"Error {e}")
                            picture_taken = False

                self.shared_state["listening"] = False
                self.model.queue.put_nowait((capture, self, picture_taken, picture if picture_taken else None))

                # Reset State
                self.buffer = ""
                self.start_index = None

    def transcription_worker(audio, result_queue):
        try:
            stt = STT()
            text = stt.transcribe(audio_data=audio)
            result_queue.put({"success": True, "text": text})
        except Exception as e:
            result_queue.put({"success": False, "error": str(e)})

    def run_transcription(self, shared_state, timeout=10):
        while shared_state["running"]:
            try:
                sleep(0.1)
                
                audio_clip = self.audio_queue.get()
                current_time = time()
                if current_time - self.listen_time < 5:
                    continue

                if shared_state["listening"]: 
                    result_queue = multiprocessing.Queue()
                    process = multiprocessing.Process(target=MirAI.transcription_worker, args=(audio_clip, result_queue))
                    process.start()
                    process.join(timeout)

                    if process.is_alive():
                        process.terminate()
                        process.join()
                        self.logger.info("Whisper timed out!")
                    
                    result = result_queue.get() if not result_queue.empty() else None

                    if result:
                        self.transcription_queue.put(result)
            except KeyboardInterrupt:
                break

    def run(self):
        print("MirAI ready!")

        while self.shared_state["running"]:
            try:
                if self.transcription_queue.qsize() > 0:
                    transciption_result = self.transcription_queue.get()
                    if transciption_result["success"]:
                        self.find_trigger(transciption_result["text"])
                    else:
                        self.logger.error(f"Error: {transciption_result['error']}")
            except KeyboardInterrupt:
                self.shared_state["running"] = False
                break

        print("Exiting normally.")

        self.listener_thread.join()
        print("Listener joined.")

        self.transcription_thread.join()
        print("Transcriber joined.")

        self.model.close()
        print("Closed llama-server")
    
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

    with multiprocessing.Manager() as manager:
        mirai = MirAI(manager, args.voice, args.microphone, args.llm, args.images, args.record, args.no_window)
        mirai.run()
import base64
import logging
import io
from PIL import Image
import json
import numpy as np
from queue import Full, Empty
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

def audio_listener(mic_name: str, audio_queue: multiprocessing.Queue, shared_state, recording: bool = False):
    if recording:
        os.makedirs('./out/recordings', exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_filename = os.path.join('./out/recordings', f'recording_{timestamp}.flac')
        file = SoundFile(output_filename, mode='w', samplerate=16000, channels=1, format='FLAC')
    
    microphone = Microphone(mic_name)

    while shared_state["running"]:
        try:
            audio_clip = microphone.record(0.5)
            
            if recording:
                file.write(audio_clip)
                file.flush()

            if shared_state["listening"]:
                try:
                    audio_queue.put_nowait(audio_clip)
                except Full:
                    shared_state["listening"] = False

        except KeyboardInterrupt:
            break

    if recording:
        file.close() 

def transcription_listener(submit_queue: multiprocessing.Queue, result_queue: multiprocessing.Queue):
    try:
        stt = STT()

        while True:
            try:
                submitted_audio, duration = submit_queue.get()
                proc_start = time()
                text = stt.transcribe(submitted_audio)
                result_queue.put_nowait(text)
                proc_end = time()

                print(f"Whisper took: {proc_end - proc_start:.3f} seconds for {duration:.3f} seconds audio; Performance is {duration / (proc_end - proc_start) * 100:.3f}% realtime")
            except Empty:
                sleep(0.5)
                continue
    except KeyboardInterrupt:
        pass

class MirAI:
    def __init__(self, manager, voice_path: str, microphone: str, llm_path: str, record_mode: bool = False, config_path: str = 'config.json'):
        self.logger =logging.getLogger("MirAI")

        self.shared_state = manager.dict()
        self.shared_state["running"] = True
        self.shared_state["listening"] = False
        self.transcribed = ""
        self.response_buffer = ""

        self.audio_queue = multiprocessing.Queue(maxsize=60)
        self.listener_thread = multiprocessing.Process(target=audio_listener, args=(microphone, self.audio_queue, self.shared_state, record_mode))

        self.transcription_queue = multiprocessing.Queue(maxsize=1)        

        self.task1q = multiprocessing.Queue(maxsize=1)
        self.task2q = multiprocessing.Queue(maxsize=1)

        self.task1_thread = multiprocessing.Process(target=transcription_listener, args=(self.task1q, self.transcription_queue))
        self.task2_thread = multiprocessing.Process(target=transcription_listener, args=(self.task2q, self.transcription_queue))

        self.invoker_thread = Thread(target=MirAI.invoker_thread, args=(self,))

        default_picture_phrases = ["take a picture"]

        base_host = "127.0.0.1"
        base_port = 8000

        cheat_host = "1.1.1.1"
        cheat_port = 8000

        prompt = "You are a useful AI assistant. Please help the user."

        try:
            with open(config_path, 'r') as file:
                data = json.load(file)
            
            self.picture_strings = data.get('picture_phrases', default_picture_phrases)

            base_host = data["base_host"]
            base_port = data["base_port"]

            cheat_host = data["cheat_host"]
            cheat_port = data["cheat_port"]

            prompt = data["prompt"]
        except (FileNotFoundError, json.JSONDecodeError) as e:
            self.logger.error(f"Failed to load JSON File: {e}")
            self.picture_strings = default_picture_phrases

        self.model = Model(llm_path, voice_path, self, prompt=prompt, cheat_host=cheat_host, cheat_port=cheat_port, host=base_host, port=base_port)
        
        # Start threads at end
        self.listener_thread.start()
        self.task1_thread.start()
        self.task2_thread.start()
        
        self.active_task_queue = self.task1q
        self.active_thread = self.task1_thread
        self.backup_task_queue = self.task2q
        self.backup_thread = self.task2_thread
        
        self.invoker_thread.start()

    def transcribe_submit(self, submit_data):
        # Submit with timeout
        self.active_task_queue.put(submit_data)
        start_time = time()
        got_result = False

        # Wait for first result
        while time() - start_time < 20:
            try:
                result = self.transcription_queue.get(timeout=1)
                got_result = True
                break
            except Empty:
                continue
                    
        if not got_result:
            self.backup_task_queue.put(submit_data)
            start_time = time()

            # Now restart the secondary process                        
            self.active_thread.terminate()
            self.active_thread.join()

            self.active_thread = multiprocessing.Process(target=transcription_listener, args=(self.task1q, self.transcription_queue))
            self.active_thread.start()

            # Swap our "active" and "background" while Whisper loads
            self.active_task_queue, self.backup_task_queue = self.backup_task_queue, self.active_task_queue
            self.active_thread, self.backup_thread = self.backup_thread, self.active_thread

            # Wait for results
            while time() - start_time < 20:
                try:
                    result = self.transcription_queue.get(timeout=1)
                    got_result = True
                    break
                except Empty:
                    continue

            if not got_result:
                raise RuntimeError("Whisper failed twice somehow!")
        
        return result


    def invoker_thread(self):
        print("MirAI starting...")

        while self.shared_state["running"]:
            try:
                sleep(0.1)
                if not self.shared_state["listening"] and self.audio_queue.qsize() > 0:
                    # Obtain all audio, concat into a buffer, submit
                    array_of_arrays = []
                    for _ in range(0, self.audio_queue.qsize()):
                        array_of_arrays.append(self.audio_queue.get_nowait())
                    
                    final_buffer = np.concatenate(array_of_arrays)
                    duration = 0.5 * len(array_of_arrays)

                    result = self.transcribe_submit((final_buffer, duration))

                    picture_taken = False
                    picture = None
                    
                    # Check for camera phrase
                    for sub in self.picture_strings:
                        if sub.lower() in result.lower():
                            picture_taken, picture = self.take_picture()
                    

                    # Submit to LLM
                    self.transcribed = result
                    self.response_buffer = ""
                    print(result)
                    self.model.queue.put_nowait((result, self, picture_taken, picture if picture_taken else None))

            except (KeyboardInterrupt, BrokenPipeError):
                break
        
        self.active_thread.terminate()
        self.active_thread.join()

        self.backup_thread.terminate()
        self.backup_thread.join()

        self.model.close()
        
    def take_picture(self):
        try:
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                self.model.tts.say("Could not open camera!")
                return (False, None)

            cv2.waitKey(5000)
            self.model.tts.say("Taking picture now!")

            ret, frame = cap.read()

            if not ret:
                self.model.tts.say("Could not capture frame!")
                return (False, None)


            image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            buffer = io.BytesIO()
            image.save(buffer, format = "PNG")
            picture = base64.b64encode(buffer.getvalue()).decode('utf-8')
            
            return (True, picture)
        except Exception as e:
            print(e)
            return (False, None)

    def toggle_listening(self):
        self.shared_state["listening"] = not self.shared_state["listening"]

    def toggle_cheat_mode(self):
        self.logger.info(f"Cheat Mode Set: {not self.model.cheat_mode}")
        self.model.cheat_mode = not self.model.cheat_mode

def main():
    logging.basicConfig(filename="mirai.log", level=logging.DEBUG)
    logging.getLogger("faster_whisper").setLevel(logging.DEBUG)

    parser = ArgumentParser(description="Start Robot Waifu Program")
    parser.add_argument('-c', '--config', type=str, default="config.json", help="Default configuration file")
    parser.add_argument('-i', '--images', type=str, default="images", help="Path to images directory for display")
    parser.add_argument('-l', '--llm', required=True, type=str, help="Model path (.gguf)")
    parser.add_argument('-m', '--microphone', required=True, type=str, help="Micrpohone Name")
    parser.add_argument('-r', '--record', action='store_true', help="Record microphone and video")
    parser.add_argument('-v', '--voice', required=True, type=str, help="Voice Model File (.onnx)")

    args = parser.parse_args()

    with multiprocessing.Manager() as manager: 
        mirai = MirAI(manager, args.voice, args.microphone, args.llm, args.record, args.config)
        
        ui = UI()        
        ui.run_customtkinter(mirai, args.images, "MirAI")
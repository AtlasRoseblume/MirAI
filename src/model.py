import logging
from subprocess import Popen, STDOUT
from time import time, sleep
from threading import Thread
from queue import Queue
from .tts import TTS

class Model:
    def __init__(self, model_path: str, voice_path: str, host: str = "127.0.0.1", port: int = 8000):
        self.logger = logging.getLogger("Model")
        
        command = [
            "python3", "-m", "llama_cpp.server",
            "--model", model_path,
            "--host", host,
            "--port", str(port),
            "--chat_format", "openai"
        ]

        # Open in the background
        self.process = Popen(command, stdout=open("llama_server.log", "w"), stderr=STDOUT, start_new_session=True)
        self.logger.info(f"Started Llama server at {host} on port {port}.\nRunning model {model_path}")

        self.tts = TTS(voice_path)
        self.running = True
        self.queue = Queue()

        self.submit_thread = Thread(target=Model.submit_listener, args=(self,))
        self.submit_thread.start()


    def submit_listener(self):
        sleep(3)
        while self.running:
            if self.queue.qsize() == 0:
                sleep(1)
                continue

            user_input, core_state = self.queue.get()

            # TODO: Submit

            self.tts.say(user_input)

            # Reset core state, ready to go again
            core_state.listen_time = time()
            core_state.listening = True


    def close(self):
        self.running = False
        self.process.terminate()
        sleep(3)

        if self.process.poll() is None:
            self.process.kill()
        
        self.logger.info("Llama server killed")

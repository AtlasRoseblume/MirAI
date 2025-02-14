import logging
from subprocess import Popen, STDOUT
from time import time, sleep
from threading import Thread
from queue import Queue
from .tts import TTS

from langchain_openai import ChatOpenAI
from langchain_core.chat_history import (
    BaseChatMessageHistory,
    InMemoryChatMessageHistory,
)
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.messages import HumanMessage, AIMessage

store = {}
def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    
    return store[session_id]

class Model:
    def __init__(self, model_path: str, voice_path: str, host: str = "127.0.0.1", port: int = 8000):
        self.logger = logging.getLogger("Model")
        
        command = [
            "python3", "-m", "llama_cpp.server",
            "--model", model_path,
            "--host", host,
            "--port", str(port),
            "--n_threads", "4",
            "--n_ctx", "4096",
        ]

        # Open in the background
        self.process = Popen(command, stdout=open("llama_server.log", "w"), stderr=STDOUT, start_new_session=True)
        self.logger.info(f"Started Llama server at {host} on port {port}.\nRunning model {model_path}")

        self.tts = TTS(voice_path)
        self.running = True
        self.queue = Queue()

        self.model = ChatOpenAI(
            model="gemma",
            base_url=f"http://{host}:{port}/v1",
            api_key="not_needed",
            temperature=0.7
        )

        self.prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a helpful assistant roleplaying as Furina de Fontaine, the hydro archon of the fictional land of Fontaine. You are to answer shortly, but elegantly."
                ),
                MessagesPlaceholder(variable_name="messages")
            ]
        )

        self.chain = self.prompt | self.model

        self.base_model = RunnableWithMessageHistory(self.chain, get_session_history)
        self.config = {"configurable": {"session_id": "abc1"}}

        self.submit_thread = Thread(target=Model.submit_listener, args=(self,))
        self.submit_thread.start()

    def submit_listener(self):
        sleep(3)
        while self.running:
            if self.queue.qsize() == 0:
                sleep(1)
                continue

            user_input, core_state = self.queue.get()

            buffer = ""

            start = time()
            print("Response:")
            for r in self.base_model.stream(
                [HumanMessage(content=user_input)],
                config=self.config,
            ):
                buffer += r.content

                for i in range(len(buffer)):
                    if buffer[i] in ['.', '?', '!']:
                        print(buffer[0:i+1])
                        self.tts.say(buffer[0:i+1])
                        buffer = buffer[i + 1:]
                        break

            end = time()

            print(f"AI Response {end - start}s")
            self.tts.say(buffer)

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

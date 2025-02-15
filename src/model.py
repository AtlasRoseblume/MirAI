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
from langchain_core.messages import HumanMessage, AIMessage, AIMessageChunk

store = {}
def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()

        try:
            with open("chat_history.txt", "r") as f:
                store[session_id].messages = eval(f.read())
        except Exception as e:
            print(f"ERROR: {e}")
    
    return store[session_id]

class Model:
    def __init__(self, model_path: str, voice_path: str, core, cheat_host: str, cheat_port: int, prompt: str, host: str = "127.0.0.1", port: int = 8000):
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
            model="qwen2.5",
            base_url=f"http://{host}:{port}/v1",
            api_key="not_needed",
            temperature=0.7
        )

        self.cheat_model = ChatOpenAI(
            model="llava",
            base_url=f"http://{cheat_host}:{cheat_port}/v1",
            api_key="not_needed",
            temperature=0.7,
            timeout=10,
            max_retries=3
        )

        self.prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    prompt    
                ),
                MessagesPlaceholder(variable_name="messages")
            ]
        )

        self.chain = self.prompt | self.model
        self.cheat_chain = self.prompt | self.cheat_model

        self.base_model = RunnableWithMessageHistory(self.chain, get_session_history)
        self.cheat_model = RunnableWithMessageHistory(self.cheat_chain, get_session_history)
        self.config = {"configurable": {"session_id": "abc1"}}

        self.cheat_mode = False

        self.submit_thread = Thread(target=Model.submit_listener, args=(self, core))
        self.submit_thread.start()

    def submit_listener(self, core):
        sleep(3)
        while self.running:
            if self.queue.qsize() == 0:
                sleep(1)
                continue

            user_input, core_state, picture_taken, picture = self.queue.get()

            buffer = ""

            start = time()
            print("Response:")

            if self.cheat_mode:
                selected_model = self.cheat_model
            else:
                selected_model = self.base_model

            try:

                if picture_taken:
                    if self.cheat_mode:
                        # Add picture image processed first
                        message = HumanMessage(content=[
                            {"type": "text", "text": f"Describe the image shown and use it to complete any questions asked. {user_input}"},
                            {"type": "image_url", "image_url": f"data:image/png;base64,{picture}"}
                        ])
                        # We cannot save this history; It destroys everything
                        selected_model = self.cheat_chain
                    else:
                        self.tts.say("Cheat mode deactivated, cannot use picture data.")
                        message = HumanMessage(content=user_input)
                else:
                    message = HumanMessage(content=user_input)

                for r in selected_model.stream(
                    [message],
                    config=self.config,
                ):
                    buffer += r.content
                    core.response_buffer += r.content

                    for i in range(len(buffer)):
                        if buffer[i] in ['.', '?', '!']:
                            print(buffer[0:i+1])
                            self.tts.say(buffer[0:i+1])
                            buffer = buffer[i + 1:]
                            break
            except Exception as e:
                print(f"ERROR: {e}")

                if self.cheat_mode:
                    self.cheat_mode = False


            end = time()

            print(f"AI Response {end - start}s")
            if(len(buffer) > 1):
                self.tts.say(buffer)

            with open("chat_history.txt", "w") as f:
                print(get_session_history(self.config["configurable"]["session_id"]).messages, file=f)

            # Reset core state, ready to go again
            core_state.listen_time = time()
            core_state.shared_state["listening"] = True


    def close(self):
        self.running = False
        self.process.terminate()
        sleep(3)

        if self.process.poll() is None:
            self.process.kill()
        
        self.logger.info("Llama server killed")

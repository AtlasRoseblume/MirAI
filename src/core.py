import logging
from time import time
from argparse import ArgumentParser
from .tts import TTS
from .stt import STT

def main():
    logging.basicConfig(filename="stt.log", level=logging.DEBUG)
    logging.getLogger("faster_whisper").setLevel(logging.DEBUG)

    parser = ArgumentParser(description="Start Robot Waifu Program")
    parser.add_argument('-v', '--voice', required=True, type=str, help="Voice Model File (.onnx)")

    args = parser.parse_args()

    stt = STT("UAC 1.0")
    stt.record_transcribe()
    stt.record_transcribe()

    tts = TTS(args.voice) 
    tts.say("OOF!", 1.0)
    tts.close()
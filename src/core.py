from time import time
from argparse import ArgumentParser
from .tts import TTS

def main():
    parser = ArgumentParser(description="Start Robot Waifu Program")
    parser.add_argument('-v', '--voice', required=True, type=str, help="Voice Model File (.onnx)")

    args = parser.parse_args()

    tts = TTS(args.voice)

    print(f"Voice Model Loaded: {args.voice}")

    start = time()
    
    tts.say("RAWR UWU XD OWO MEOW Owo NYA uwU NEKO OwU-CHAN TeeHee UwU UwO Ehe~ Kawaii ~~", 1.0)
    end = time()

    print(f"Time Taken: {end - start}")
    
    tts.close()
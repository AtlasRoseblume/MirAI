# MirAI
Making A Smart Robot Waifu

## Setup

```sh
sudo apt update && sudo apt install portaudio19-dev
python3 -m venv venv ./venv/
source ./venv/bin/activate
pip install -r src/requirements.txt
piper --model <your_model_choice> --output_file test.wav # Download the model file (keep the json)
```

## Running

```sh
python3 -m src --voice /path/to/model
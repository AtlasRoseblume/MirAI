# MirAI
Making A Smart Robot Waifu

## Setup

```sh
sudo apt update && sudo apt install portaudio19-dev libraspberrypi-dev libcap-dev
python3 -m venv --system-site-packages venv ./venv/ 
source ./venv/bin/activate
pip install -r src/requirements.txt
piper --model <your_model_choice> --output_file test.wav # Download the model file (keep the json)
# Download a SLM model, for RPI 5, Gemma2, Qwen-2.5 are recommended
```

## Running

```sh
python3 -m src --voice /path/to/model --microphone "device_name" --model /path/to/model
```
import os

def get_tts():
    return os.getenv(
        "TTS_MODEL",
        "cartesia/sonic-3:9626c31c-bec5-4cca-baa8-f8ba9e84c8bc",
    )
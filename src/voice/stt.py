import os

def get_stt():
    return os.getenv("STT_MODEL", "deepgram/nova-3:multi")
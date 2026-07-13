
from langchain.chat_models import init_chat_model
import os


model = init_chat_model(
    "gpt-5.4-mini",
    temperature=0,
    api_key=os.getenv("OPENAI_API_KEY"),
)
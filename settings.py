from dotenv import load_dotenv
import os

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME")
COMPLETION_MODEL_NAME = os.getenv("COMPLETION_MODEL_NAME")
BASE_URL = os.getenv("BASE_URL")
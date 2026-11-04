import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()
def get_llm_client():
    try:
        model_name = os.getenv("LLM_DEFAULT_MODEL")
        api_key = os.getenv("OPENAI_API_KEY")
        base_url = os.getenv("OPENAI_API_BASE")
        client = ChatOpenAI(
            model_name=model_name,
            api_key = api_key,
            base_url=base_url,
            temperature=0.0,
        )
        return client
    except Exception as e:
        raise e

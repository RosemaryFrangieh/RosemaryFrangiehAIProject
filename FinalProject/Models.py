
from dotenv import find_dotenv, load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv(find_dotenv(), override=True)

fast_model = ChatOpenAI(
    model="gpt-5.4-nano",
    temperature=0,
    max_retries=1,
)

powerful_model = ChatOpenAI(
    model="gpt-5.4",
    temperature=0,
    max_retries=1,
    disable_streaming=True,
)

secondary_model = ChatOpenAI(
    model="gpt-5.4-mini",
    temperature=0,
    max_retries=1,
    disable_streaming=True,
)
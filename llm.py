import os
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

MODEL = os.getenv("GROQ_MODEL")
API_KEY = os.getenv("GROQ_API_KEY")

if not MODEL or not API_KEY:
    raise RuntimeError("Set GROQ_API_KEY and GROQ_MODEL in your .env file")

_client = Groq(api_key=API_KEY)


def model_name() -> str | None:
    return MODEL


def generate(prompt: str, temperature: float = 0.0) -> str | None:
    """Send one prompt, return the answer text. Swap providers by editing only this file."""
    resp = _client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
    )
    return resp.choices[0].message.content
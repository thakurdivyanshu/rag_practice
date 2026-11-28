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
        model=MODEL, # type: ignore
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
    )
    return resp.choices[0].message.content

def generate_structured(
    user_prompt: str,
    system_prompt: str,
    schema: dict,
    schema_name: str = "research_answer",
    temperature: float = 0.0
) -> str | None:

    resp = _client.chat.completions.create(
        model=MODEL, # type: ignore
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
            ],
        temperature=temperature,
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": schema_name,
                "strict": True,
                "schema": schema
            }
        }
    )

    return resp.choices[0].message.content
    
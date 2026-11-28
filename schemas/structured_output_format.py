from pydantic import BaseModel, ConfigDict


class ResearchAnswer(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")

    answer: str
    supported: bool
    pages: list[int]



from pydantic import BaseModel, ConfigDict

#represents one claim and if it is supported
class ClaimEvaluation(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")

    claim: str
    supported: bool

#represents all the evaluated claims
class FaithfulnessEvaluation(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")

    claims: list[ClaimEvaluation]
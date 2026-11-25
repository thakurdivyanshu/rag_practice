
import json
from pydantic import BaseModel, ConfigDict
from llm import generate, generate_structured


# 1. Define our expected output structure
class ResearchAnswer(BaseModel):
    model_config = ConfigDict(strict=False, extra="ignore")

    answer: str
    supported: bool
    pages: list[int]





prompt = """
Context:
[Page 3] Multi-head attention runs multiple attention
mechanisms in parallel to capture different relationships.

Question: What is multi-head attention?

Answer using only the context provided.

"""

print("raw llm response")
raw_response=generate(prompt)
print(raw_response)
print('+'*50)
print("Structed prompt response")
structed_response=generate_structured(prompt,schema=ResearchAnswer.model_json_schema())
structed_response=ResearchAnswer.model_validate_json(structed_response)
print(structed_response)


print("structed_response.answer :",structed_response.answer)
print("structed_response.supported :",structed_response.supported)
print("structed_response.pages :",structed_response.pages)



#bad response
# bad_response = """
# {
#     "answer": "Multi-head attention captures different relationships.",
#     "supported": "yes",
#     "pages": [3],
#     "confidence": 0.95
# }
# """
# try:
#     result=ResearchAnswer.model_validate_json(bad_response)
#     print(result)
# except Exception as e:
#     print(type(e).__name__)
#     print(e)
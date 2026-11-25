
from pydantic import BaseModel

class MyNewClass(BaseModel):
    question:str
    top_k:int

request=MyNewClass(question="What is attention?",top_k=5)
request=MyNewClass(question=123,top_k="abc")

print(request)
print(request.question)
print(request.top_k)
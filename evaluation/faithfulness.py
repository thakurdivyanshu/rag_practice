#Claim extraction and faithfulness evaluation

from llm import generate_structured
from schemas.claim_evaluation_format import FaithfulnessEvaluation
 

def evaluate_faithfulness(context,answer):
    system_prompt = """
        You are a faithfulness evaluator.

        Your task is to evaluate whether each claim in the answer
        is supported by the retrieved context.

        Instructions:
        1. Break the answer into individual, atomic claims.
        2. Evaluate every claim independently.
        3. Mark a claim as supported only if all its meaningful details
        are supported by the context.
        4. Mark a claim as unsupported if any meaningful detail is
        missing or contradicted by the context.
        5. Use only the retrieved context as evidence.
        6. Do not use outside knowledge.
        7. Return every claim, including unsupported claims.
        8. Keep claims as small as possible. Do not combine independent facts into a single claim.
    """

    user_prompt = f"""
    retrieved_context:
    {context}

    generated_answer:
    {answer}
    """

    schema=FaithfulnessEvaluation.model_json_schema()

    response = generate_structured(
        user_prompt=user_prompt,
        system_prompt=system_prompt,
        schema=schema,
        schema_name="faithfulness_evaluation"
    )

    result= FaithfulnessEvaluation.model_validate_json(response) # type: ignore
    score = sum(claim.supported for claim in result.claims)
    if not result.claims:
        raise ValueError("Faithfulness evaluator returned no claims")
    faithfulness_score=score/len(result.claims)
    print(f"Faithfulness: {faithfulness_score * 100:.2f}%")

    print(f"{'+'*50} claims {'+'*50}\n")
    print(result.claims)

    return result, faithfulness_score
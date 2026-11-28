def reciprocal_rank(expected_pages,retrieved_pages):
    for rank,page in enumerate(retrieved_pages,start=1):
        if page in expected_pages:
            return 1/rank

    return 0.0

#Hit@K tells us whether we found at least one relevant page
def hit_at_k(expected_page,retrieved_pages):
    return int(any(page in expected_page for page in retrieved_pages))

# Recall@K measures what % of the expected relevant pages we actually retrieved
def recall_at_k(expected_pages,retrieved_pages):
    if not expected_pages:
        return 0.0

    relevent_retrieved = set(expected_pages) & set(retrieved_pages)
    return len(relevent_retrieved)/len(set(expected_pages))


def mean_reciprocal_rank(evaluation_results):
    rr_scores = []
    if not evaluation_results:
        return 0.0
    for expected_pages, retrieved_pages in evaluation_results:
        rr_scores.append(
            reciprocal_rank(expected_pages, retrieved_pages)
        )

    return sum(rr_scores) / len(rr_scores)

 
 
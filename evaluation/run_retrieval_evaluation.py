from evaluation.eval_dataset import EVAL_DATASET
from evaluation.retrieval_metrics import hit_at_k, mean_reciprocal_rank, recall_at_k, reciprocal_rank
from rag_test import retrieve_chunks

evaluation_results=[]
hit_scores = []
recall_scores = []

for evals in EVAL_DATASET:
    
    #only for IN questoin
    if evals["category"] != "IN":
        continue

    print(f"question: {evals['question']}\n")

    retrieved_chunks=retrieve_chunks(evals["question"])
    retrieved_pages=[doc.metadata.get("page") for doc,score in retrieved_chunks]
    print("Retrieved pages:", retrieved_pages)
    print("Unique pages:", list(dict.fromkeys(retrieved_pages)))
    retrieved_pages=list(dict.fromkeys(retrieved_pages))
    expected_pages=evals["expected_pages"]

    print('+'*10,"reciprocal_rank",'*'*10)
    print(reciprocal_rank(expected_pages=expected_pages,retrieved_pages=retrieved_pages))

    print('+'*10,"hit_at_k",'*'*10)
    print(hit_at_k(expected_pages,retrieved_pages))

    print('+'*10,"recall_at_k",'*'*10)
    print(recall_at_k(expected_pages,retrieved_pages))  

    evaluation_results.append((expected_pages,retrieved_pages))

    hit_score = hit_at_k(expected_pages, retrieved_pages)
    recall_score = recall_at_k(expected_pages, retrieved_pages)

    hit_scores.append(hit_score)
    recall_scores.append(recall_score)

    print("="*50)

print('+'*10,"mean_reciprocal_rank",'*'*10)
print(mean_reciprocal_rank(evaluation_results))  
hit_rate = sum(hit_scores) / len(hit_scores)
average_recall = sum(recall_scores) / len(recall_scores)

print('+'*10,"Hit Rate@4",'*'*10)
print("Hit Rate@4:", hit_rate)

print('+'*10,"Average Recall@4",'*'*10)
print("Average Recall@4:", average_recall)

# MRR = 0.90: On average, the first relevant page appears very high in the search results (closer to rank 1).



# Hit Rate@4 = 100%, Average Recall@4 = 95%: Every question found at least one relevant page, and we retrieved 95% of all expected relevant pages on average.
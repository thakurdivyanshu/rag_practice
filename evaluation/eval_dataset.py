EVAL_DATASET = [
    # IN: The answer is available in the paper.
    {
        "id": "Q01",
        "question": "What is multi-head attention?",
        "category": "IN",
        "expected_pages": [3, 4]
    },
    {
        "id": "Q02",
        "question": "How is scaled dot-product attention calculated?",
        "category": "IN",
        "expected_pages": [3]
    },
    {
        "id": "Q03",
        "question": "Why do the authors divide the query-key dot products by the square root of the key dimension?",
        "category": "IN",
        "expected_pages": [3]
    },
    {
        "id": "Q04",
        "question": "How many layers does the Transformer encoder contain, and what are its two sub-layers?",
        "category": "IN",
        "expected_pages": [2]
    },
    {
        "id": "Q05",
        "question": "What additional sub-layer does the Transformer decoder have compared with the encoder?",
        "category": "IN",
        "expected_pages": [2]
    },
    {
        "id": "Q06",
        "question": "Why does the Transformer use positional encodings, and which functions are used to create them?",
        "category": "IN",
        "expected_pages": [4]
    },
    {
        "id": "Q07",
        "question": "What is the structure of the position-wise feed-forward network?",
        "category": "IN",
        "expected_pages": [4]
    },
    {
        "id": "Q08",
        "question": "Which optimizer did the authors use, and what values did they use for beta1 and beta2?",
        "category": "IN",
        "expected_pages": [6]
    },
    {
        "id": "Q09",
        "question": "What training datasets did the authors use for English-to-German and English-to-French translation?",
        "category": "IN",
        "expected_pages": [7]
    },
    {
        "id": "Q10",
        "question": "According to Table 2, what BLEU scores did the Transformer big model achieve on the English-to-German and English-to-French tasks?",
        "category": "IN",
        "expected_pages": [7]
    },

    # TRAP: Related to the paper, but the specific answer is not provided.
    {
        "id": "Q11",
        "question": "What BLEU score did the Transformer achieve on the WMT 2014 German-to-English translation task?",
        "category": "TRAP",
        "expected_pages": []
    },
    {
        "id": "Q12",
        "question": "What is the maximum context window length supported by the Transformer described in this paper?",
        "category": "TRAP",
        "expected_pages": []
    },
    {
        "id": "Q13",
        "question": "How many trainable parameters are present specifically in the Transformer decoder?",
        "category": "TRAP",
        "expected_pages": []
    },
    {
        "id": "Q14",
        "question": "What dataset did the authors use to train the Transformer for English-to-Spanish translation?",
        "category": "TRAP",
        "expected_pages": []
    },

    # OUT: Completely unrelated to the paper.
    {
        "id": "Q15",
        "question": "What is the capital of France?",
        "category": "OUT",
        "expected_pages": []
    },
    {
        "id": "Q16",
        "question": "How many parameters does GPT-4 have?",
        "category": "OUT",
        "expected_pages": []
    },
    {
        "id": "Q17",
        "question": "How do I make sourdough bread at home?",
        "category": "OUT",
        "expected_pages": []
    },
    {
        "id": "Q18",
        "question": "Who won the FIFA World Cup in 2022?",
        "category": "OUT",
        "expected_pages": []
    }
]
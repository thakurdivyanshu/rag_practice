Step 4: History-aware retrieval
WHY do we need this?
Our current pipeline has one limitation:

Imagine this conversation:
- You: What is self-attention?
- Chatbot: Self-attention helps tokens understand relationships with other tokens.
- You: How does it work?
Our LLM understands that "it" means self-attention because it receives chat history.

But FAISS receives only:

How does it work?

FAISS doesn't understand the conversation. It searches for chunks similar to that incomplete question.

Our solution: Before sending the question to FAISS, we'll use the LLM to rewrite it as a standalone question.
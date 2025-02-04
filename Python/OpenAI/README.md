# OpenAI

## Temperature

Lower temperature (0.1–0.3): Deterministic outputs, ideal for factual responses.
Higher temperature (0.7+): More diverse and creative responses. Example:

## Tokens

### What Are Tokens?

Tokens are the basic units of text that the model processes. A token can be as short as one character or as long as one word:

- "ChatGPT" → 2 tokens
- "Hello, world!" → 4 tokens ("Hello", ",", "world", "!")
- Spaces and punctuation count as tokens.

### Counting Tokens

Token usage includes both input and output. If your input prompt is 1,000 tokens and the output is 500 tokens, your total usage is 1,500 tokens.

## Embeddings

### What Are Embeddings?

Embeddings are numerical vector representations of text. They capture the semantic meaning of text, allowing for tasks like:

- Text similarity comparisons
- Document search and clustering
- Classification and sentiment analysis

## Embeddings Vs. Tokens

| **Feature**    | **Tokens**                                            | **Embeddings**                                      |
| -------------- | ----------------------------------------------------- | --------------------------------------------------- |
| **Definition** | Smallest units of text processed by the model.        | Numerical vectors representing the meaning of text. |
| **Purpose**    | Enable the model to generate or understand text.      | Capture semantic relationships and meanings.        |
| **Output**     | Text (split into tokens).                             | High-dimensional numerical arrays (vectors).        |
| **Use Cases**  | Input/output of language models, pricing, and limits. | Similarity search, clustering, semantic tasks.      |
| **Scope**      | Operates on text as it is written.                    | Operates on the underlying meaning of text.         |

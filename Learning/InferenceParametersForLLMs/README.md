# Inference Parameters for LLM

### 1. **Temperature**

- **Definition**: Controls the randomness of the predictions.
- **Range**: 0.0 to 1.0 (sometimes higher values are supported).
- **How It Works**:
  - Lower values (e.g., 0.1) make the model more deterministic, favoring the most probable next words.
  - Higher values (e.g., 0.9) increase randomness, making outputs more diverse but less predictable.
- **Use Case**: Use low temperature for tasks requiring precision (e.g., factual answers) and high temperature for creative tasks (e.g., storytelling).

---

### 2. **Top-k Sampling**

- **Definition**: Limits the model to selecting the next word only from the top-k most probable options.
- **How It Works**:
  - If \( k = 50 \), the model considers only the 50 most probable tokens for each prediction.
  - Larger \( k \) values allow broader choices; smaller values make the model more focused.
- **Use Case**: Balances diversity and relevance when generating responses.

---

### 3. **Top-p Sampling (Nucleus Sampling)**

- **Definition**: Selects from a dynamic set of top tokens whose cumulative probability is below a threshold \( p \).
- **How It Works**:
  - If \( p = 0.9 \), the model chooses from the smallest set of tokens that account for 90% of the probability distribution.
  - Combines the benefits of diversity (like high \( k \)) with control over randomness.
- **Use Case**: Preferred over top-k for more natural and context-sensitive outputs.

---

### 4. **Max Tokens**

- **Definition**: Sets the maximum number of tokens the model can generate in a single response.
- **How It Works**:
  - Ensures responses fit within specific length constraints.
  - Tokens include words, punctuation, and formatting symbols.
- **Use Case**: Limit response length for concise answers or match output to a required format.

---

### 5. **Frequency Penalty**

- **Definition**: Penalizes tokens that have already appeared in the generated text.
- **How It Works**:
  - Encourages the model to use less repetitive language by assigning a penalty to frequently used tokens.
- **Use Case**: Reduces redundancy in tasks like summarization or creative writing.

---

### 6. **Presence Penalty**

- **Definition**: Penalizes tokens that have already been used in the context (not just the generated text).
- **How It Works**:
  - Promotes diversity by discouraging over-reliance on specific words or phrases already present in the input.
- **Use Case**: Encourages novelty, especially in brainstorming or generating diverse ideas.

---

### 7. **Stop Sequences**

- **Definition**: Specifies one or more token sequences where the model should stop generating.
- **How It Works**:
  - The model halts generation when a stop sequence is encountered.
- **Use Case**: Useful for tasks requiring structured outputs, such as JSON or delimited text.

---

### 8. **Beam Search (Advanced)**

- **Definition**: A deterministic search algorithm that evaluates multiple potential sequences and selects the best one.
- **How It Works**:
  - Generates multiple sequences in parallel and ranks them by a scoring function.
  - May lead to better optimization but can reduce randomness.
- **Use Case**: Suitable for translation or summarization where coherence is critical.

---

### Choosing Parameters

- For **deterministic outputs**: Use low temperature (e.g., 0.1) and no sampling.
- For **creative outputs**: Use high temperature (e.g., 0.8–1.0) with top-p sampling (e.g., \( p = 0.9 \)).
- For **controlled length**: Set max tokens and use stop sequences.
- For **diversity**: Combine frequency penalty and presence penalty with top-p or top-k sampling.

These parameters can be combined to suit specific applications, providing flexibility in tailoring LLM responses to user needs.

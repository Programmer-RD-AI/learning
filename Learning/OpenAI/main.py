import openai

# Set API key
openai.api_key = "your_api_key_here"  # Or use an environment variable

response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",  # Or "gpt-4"
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What's the weather today?"},
    ],
    temperature=0.7,
)

print(response["choices"][0]["message"]["content"])

response = openai.Completion.create(
    model="text-davinci-003",
    prompt="Write a short poem about the sea.",
    max_tokens=50,
    temperature=0.7,
)

print(response["choices"][0]["text"].strip())

response = openai.Embedding.create(
    model="text-embedding-ada-002",
    input="OpenAI embeddings are useful for many applications.",
)

print(response["data"][0]["embedding"])

response = openai.File.create(file=open("my_data.jsonl", "rb"), purpose="fine-tune")

print(response)

# {"prompt": "Translate to French: Hello!", "completion": "Bonjour!"}
# openai api fine_tunes.create -t "data.jsonl" -m "davinci"

try:
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo", messages=[{"role": "user", "content": "Tell me a joke!"}]
    )
    print(response["choices"][0]["message"]["content"])
except openai.error.OpenAIError as e:
    print(f"Error: {e}")


import boto3
import json

# Initialize the Bedrock client
bedrock = boto3.client(
    "bedrock",
    region_name="us-east-1",
)

# Model and input
model_id = "anthropic.claude-v1"
input_text = "What is the capital of France?"

# Call the Converse API
try:
    response = bedrock.invoke_model(
        modelId=model_id,
        body=input_text,
        accept="application/json",
        contentType="application/json",
    )
    # Parse and display response
    response_text = json.loads(response["body"].read().decode("utf-8"))["result"]
    print("Model Response:", response_text)

except Exception as e:
    print("Error:", str(e))

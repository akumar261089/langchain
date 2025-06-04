import os
from openai import AzureOpenAI
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Initialize the FastAPI app
app = FastAPI()

# Azure OpenAI settings
endpoint = "https://bh-in-openai-synapsesynergy.openai.azure.com/"
model_name = "gpt-35-turbo"
deployment = "gpt-35-turbo-2"
subscription_key = os.getenv("AZURE_API_KEY")  # Reading API key from environment variable
api_version = "2024-12-01-preview"

# Check if the environment variable is set
if subscription_key is None:
    raise ValueError("Azure API key is missing. Please set the AZURE_API_KEY environment variable.")

# Initialize the Azure OpenAI client
client = AzureOpenAI(
    api_version=api_version,
    azure_endpoint=endpoint,
    api_key=subscription_key,
)

class ChatRequest(BaseModel):
    user_message: str

@app.post("/chat")
async def chat(request: ChatRequest):
    try:
        # Get the response from OpenAI model
        response = client.chat.completions.create(
            stream=True,
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant."
                },
                {
                    "role": "user",
                    "content": request.user_message
                }
            ],
            max_tokens=4096,
            temperature=1.0,
            top_p=1.0,
            model=deployment,
        )
        
        # Collect the response
        response_content = ""
        for update in response:
            if update.choices:
                response_content += update.choices[0].delta.content or ""
        
        return {"response": response_content}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.on_event("shutdown")
def shutdown_event():
    client.close()

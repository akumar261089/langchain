import os
import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from openai import AzureOpenAI
from bs4 import BeautifulSoup

# Initialize FastAPI app
app = FastAPI()

# Azure OpenAI settings
endpoint = "https://bh-in-openai-synapsesynergy.openai.azure.com/"
model_name = "gpt-35-turbo"
deployment = "gpt-35-turbo-2"
subscription_key = os.getenv("AZURE_API_KEY")
api_version = "2024-12-01-preview"

# Check if the environment variable is set
if subscription_key is None:
    raise ValueError("Azure API key is missing. Please set the AZURE_API_KEY environment variable.")

# Initialize Azure OpenAI client
client = AzureOpenAI(
    api_version=api_version,
    azure_endpoint=endpoint,
    api_key=subscription_key,
)

# Web Search API Key (Assuming Bing API or Google Custom Search API)
SEARCH_API_KEY = os.getenv("SEARCH_API_KEY")  # Set your search API key (e.g., Bing or Google Custom Search)
SEARCH_ENGINE_ID = os.getenv("SEARCH_ENGINE_ID")  # Set your custom search engine ID

class ChatRequest(BaseModel):
    user_message: str

def perform_web_search(query: str):
    """
    Perform web search using Bing or Google Custom Search API.
    """
    search_url = "https://api.bing.microsoft.com/v7.0/search"  # Example for Bing
    headers = {"Ocp-Apim-Subscription-Key": SEARCH_API_KEY}
    params = {"q": query, "count": 10}
    
    response = requests.get(search_url, headers=headers, params=params)
    
    if response.status_code == 200:
        search_results = response.json()
        # Return URLs from search results
        return [item['url'] for item in search_results['webPages']['value']]
    else:
        raise Exception(f"Search API request failed with status code {response.status_code}")

def scrape_website(url: str):
    """
    Scrapes the given website URL for specific details using BeautifulSoup.
    """
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    
    # Replace with your actual scraping logic to extract the necessary fields
    details = soup.find('div', {'class': 'specific-class'})  # Placeholder for actual scraping logic
    
    if details:
        return details.text.strip()
    else:
        return None

def get_required_detail(query: str):
    """
    Uses the Web Search and Scraper Agents to find the required detail.
    If not found, perform another search.
    """
    urls = perform_web_search(query)
    
    # Try scraping each URL
    for url in urls:
        scraped_data = scrape_website(url)
        if scraped_data:
            return scraped_data
    
    # If the required detail is not found, perform another search (recursive call)
    return get_required_detail(query)

@app.post("/chat")
async def chat(request: ChatRequest):
    try:
        # Get the response from OpenAI model
        response = client.chat.completions.create(
            stream=True,
            messages=[{
                "role": "system",
                "content": "You are a helpful assistant that can search the web and scrape websites to gather information."
            }, {
                "role": "user",
                "content": request.user_message
            }],
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

        # Assume the user's message is a query for information
        query = request.user_message
        result = get_required_detail(query)

        return {"response": result}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.on_event("shutdown")
def shutdown_event():
    client.close()


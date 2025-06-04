import os
import logging
import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from openai import AzureOpenAI
from bs4 import BeautifulSoup
from urllib.parse import quote, unquote, urlparse, parse_qs
import re
from langgraph.graph import StateGraph, END
from langchain.tools import Tool
from langchain_openai import AzureChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from pydantic import BaseModel
from typing import List, Dict, Any, TypedDict
import traceback

# ========== 🔧 Logging Configuration ==========
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ========== 🔧 State Classes ==========
class OfferState(TypedDict):
    tenantName: str
    tenantDetails: str
    offerForm: List[Any]
    offerType: str
    existingOffers: List[str]
    metadata: str
    query: str
    urls: List[str]
    extracted_data: Dict[str, Any]
    llm_response: str
    response: Dict[str, Any]

# Initialize FastAPI app
app = FastAPI(title="Offer Management API", version="1.0.0")

# Azure OpenAI settings
endpoint = "https://bh-in-openai-synapsesynergy.openai.azure.com/"
model_name = "gpt-35-turbo"
deployment = "gpt-35-turbo-2"
subscription_key = os.getenv("AZURE_API_KEY")
api_version = "2024-12-01-preview"

logger.info("Initializing Azure OpenAI client...")

if subscription_key is None:
    logger.error("Azure API key is missing. Please set the AZURE_API_KEY environment variable.")
    raise ValueError("Azure API key is missing. Please set the AZURE_API_KEY environment variable.")

try:
    client = AzureOpenAI(
        api_version=api_version,
        azure_endpoint=endpoint,
        api_key=subscription_key,
    )
    
    llm = AzureChatOpenAI(
        azure_deployment=deployment,
        azure_endpoint=endpoint,
        api_key=subscription_key,
        api_version=api_version,
        temperature=0.7
    )
    logger.info("Azure OpenAI client initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize Azure OpenAI client: {str(e)}")
    raise

class ChatRequest(BaseModel):
    tenantName: str
    tenantDetails: str
    offerForm: List[Any]
    offerType: str
    existingOffers: List[str]
    metadata: str

# ========== 🔧 Tool Functions ==========

def perform_web_search(query: str) -> List[str]:
    """Perform web search using DuckDuckGo"""
    logger.info(f"Performing web search for: {query}")
    try:
        encoded_query = quote(query.replace('"',''))
        search_url = f"https://html.duckduckgo.com/html?q={encoded_query}"
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        logger.info(f"Search url {search_url}")
        response = requests.get(search_url, headers=headers, timeout=100)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, 'html.parser')
        results = []
        
        for link in soup.find_all('a', class_='result__a', href=True):
            actual_url = extract_actual_urls(link['href'])
            if actual_url:
                results.append(actual_url)
                
        logger.info(f"Found {len(results)} search results")
        return results[:10]  # Limit to top 10 results
        
    except requests.exceptions.RequestException as e:
        logger.error(f"Web search failed: {str(e)}")
        return []
    except Exception as e:
        logger.error(f"Unexpected error in web search: {str(e)}")
        return []

def scrape_url_content(url: str) -> Dict[str, Any]:
    """Scrape content from a given URL"""
    logger.info(f"Scraping content from: {url}")
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Extract title
        title = soup.title.string.strip() if soup.title and soup.title.string else "No Title"
        
        # Extract meta description
        meta_desc = soup.find('meta', attrs={'name': 'description'})
        meta_desc_content = meta_desc.get('content', '').strip() if meta_desc else "No Meta Description"
        
        # Extract paragraphs
        paragraphs = []
        for p in soup.find_all('p')[:5]:  # Get first 5 paragraphs
            text = p.get_text().strip()
            if text and len(text) > 20:  # Only include substantial paragraphs
                paragraphs.append(text)
        
        # Extract links
        links = []
        for link in soup.find_all('a', href=True)[:10]:  # Limit to 10 links
            href = link['href']
            if re.match(r'^https?://', href):
                links.append(href)
        
        # Extract images
        images = []
        for img in soup.find_all('img', src=True)[:5]:  # Limit to 5 images
            src = img['src']
            if re.match(r'^https?://', src):
                images.append(src)
        
        result = {
            'url': url,
            'title': title,
            'meta_description': meta_desc_content,
            'paragraphs': paragraphs,
            'links': links,
            'images': images
        }
        
        logger.info(f"Successfully scraped content from {url}")
        return result
        
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to retrieve content from {url}: {str(e)}")
        return {'error': f"Failed to retrieve content: {str(e)}", 'url': url}
    except Exception as e:
        logger.error(f"Unexpected error scraping {url}: {str(e)}")
        return {'error': f"Unexpected error: {str(e)}", 'url': url}

def extract_actual_urls(redirect_url: str) -> str:
    """Extract actual URL from DuckDuckGo redirect URL"""
    try:
        parsed_url = urlparse(redirect_url)
        query_params = parse_qs(parsed_url.query)
        if 'uddg' in query_params:
            return unquote(query_params['uddg'][0])
        return None
    except Exception as e:
        logger.error(f"Error extracting URL from redirect: {str(e)}")
        return None

def extract_domain(url: str) -> str:
    """Extract domain from URL"""
    try:
        parsed = urlparse(url)
        return parsed.netloc
    except Exception as e:
        logger.error(f"Error extracting domain from {url}: {str(e)}")
        return ""

# ========== 🔧 LangChain Tools ==========

web_search_tool = Tool(
    name="DuckDuckGoSearch",
    func=perform_web_search,
    description="Performs a web search using DuckDuckGo and returns a list of URLs for the given query."
)

web_scraper_tool = Tool(
    name="WebScraper",
    func=scrape_url_content,
    description="Scrapes a given URL for title, meta description, paragraphs, links, and images."
)

# ========== 🔧 LangGraph Nodes (Agents) ==========

def agent_search_and_scrape(state: OfferState) -> OfferState:
    """Search for relevant URLs and scrape content"""
    logger.info("Starting search and scrape agent")
    
    try:
        # Create prompt
        prompt = f"""Create proper search string to get similar companies or offers for {state['offerType']} {state['tenantName']} offers {state['tenantDetails']}

Important : response should be only a short search string(max 5 words) which will give competiers with similar offers
"""
        
        # Get LLM response
        response = llm.invoke(prompt)
        search_query= response.content if hasattr(response, 'content') else str(response)
        
        logger.info(f"""LLM task agent completed successfully with response search query {search_query}""")

        # Create search query based on the request
        #search_query = f"{state['offerType']} {state['tenantName']} offers {state['tenantDetails']}"
        state['query'] = search_query
        
        # Perform web search
        urls = perform_web_search(search_query)
        state['urls'] = urls
        
        # Scrape content from top URLs
        extracted_data = {}
        for i, url in enumerate(urls[:3]):  # Scrape top 3 URLs
            logger.info(f"Scraping URL {i+1}/3: {url}")
            content = scrape_url_content(url)
            extracted_data[f"source_{i+1}"] = content
        
        state['extracted_data'] = extracted_data
        logger.info("Search and scrape agent completed successfully")
        
    except Exception as e:
        logger.error(f"Error in search and scrape agent: {str(e)}")
        state['extracted_data'] = {}
    
    return state

def agent_llm_task(state: OfferState) -> OfferState:
    """Generate response using LLM based on scraped data"""
    logger.info("Starting LLM task agent")
    
    try:
        # Prepare context from scraped data
        context = ""
        if state.get('extracted_data'):
            for source_name, data in state['extracted_data'].items():
                if isinstance(data, dict) and 'title' in data:
                    context += f"\n{source_name}:\n"
                    context += f"Title: {data.get('title', 'N/A')}\n"
                    context += f"Description: {data.get('meta_description', 'N/A')}\n"
                    if data.get('paragraphs'):
                        context += f"Content: {' '.join(data['paragraphs'][:2])}\n"
        
        # Create prompt
        prompt = f"""
        Tenant: {state.get('tenantName')}
        Details: {state.get('tenantDetails')}
        Offer Type: {state.get('offerType')}
        Existing Offers: {', '.join(state.get('existingOffers', []))}
        Metadata: {state.get('metadata')}
        
        Based on the following web search results about credit card offers:
        {context}
        
        Please provide a comprehensive summary of the best no-cost EMI offers for credit cards from UK banks.
        Focus on:
        1. Available offers and their terms
        2. Eligibility criteria
        3. Benefits and features
        4. How they compare to existing offers
        
        Provide a clear, structured response.
        """
        
        # Get LLM response
        response = llm.invoke(prompt)
        state["llm_response"] = response.content if hasattr(response, 'content') else str(response)
        
        logger.info(f"""LLM task agent completed successfully with response {state["llm_response"]}""")
        
    except Exception as e:
        logger.error(f"Error in LLM task agent: {str(e)}")
        logger.error(traceback.format_exc())
        state["llm_response"] = f"Error generating response: {str(e)}"
    
    return state

def agent_consolidate_response(state: OfferState) -> OfferState:
    """Consolidate final response"""
    logger.info("Starting response consolidation")
    
    try:
        final_response = {
            "tenantName": state.get("tenantName"),
            "offerType": state.get("offerType"),
            "searchQuery": state.get("query"),
            "sourcesFound": len(state.get("urls", [])),
            "summary": state.get("llm_response"),
            "timestamp": str(pd.Timestamp.now()) if 'pd' in globals() else "N/A"
        }
        
        state["response"] = final_response
        logger.info("Response consolidation completed successfully")
        
    except Exception as e:
        logger.error(f"Error in response consolidation: {str(e)}")
        state["response"] = {"error": f"Failed to consolidate response: {str(e)}"}
    
    return state

# ========== 🔧 LangGraph Setup ==========

logger.info("Setting up LangGraph workflow")

try:
    # Create the graph
    graph = StateGraph(OfferState)
    
    # Add nodes
    graph.add_node("SearchAndScrape", agent_search_and_scrape)
    graph.add_node("LLMTask", agent_llm_task)
    graph.add_node("ConsolidateResponse", agent_consolidate_response)
    
    # Set up the flow
    graph.set_entry_point("SearchAndScrape")
    graph.add_edge("SearchAndScrape", "LLMTask")
    graph.add_edge("LLMTask", "ConsolidateResponse")
    graph.add_edge("ConsolidateResponse", END)
    
    # Compile the graph
#    runnable = graph.compile(checkpointer=MemorySaver())
    runnable = graph.compile()
    logger.info("LangGraph workflow setup completed successfully")
    
except Exception as e:
    logger.error(f"Error setting up LangGraph workflow: {str(e)}")
    raise

# ========== 🔧 FastAPI Endpoints ==========

@app.get("/")
async def root():
    """Health check endpoint"""
    return {"message": "Offer Management API is running", "status": "healthy"}

@app.get("/health")
async def health_check():
    """Detailed health check"""
    return {
        "status": "healthy",
        "azure_openai": "configured" if subscription_key else "not configured",
        "timestamp": str(pd.Timestamp.now()) if 'pd' in globals() else "N/A"
    }

@app.post("/chat")
async def chat(request: ChatRequest):
    """Main chat endpoint for offer management"""
    logger.info(f"Received chat request for tenant: {request.tenantName}")
    
    try:
        # Validate request
        if not request.tenantName or not request.offerType:
            logger.warning("Invalid request: missing required fields")
            raise HTTPException(status_code=400, detail="tenantName and offerType are required")
        
        # Prepare initial state
        initial_state = OfferState(
            tenantName=request.tenantName,
            tenantDetails=request.tenantDetails,
            offerForm=request.offerForm,
            offerType=request.offerType,
            existingOffers=request.existingOffers,
            metadata=request.metadata,
            query="",
            urls=[],
            extracted_data={},
            llm_response="",
            response={}
        )
        
        logger.info("Invoking LangGraph workflow")
        
        # Execute the workflow
        result = runnable.invoke(initial_state)
        
        logger.info("Workflow completed successfully")
        
        return {"response": result.get("response"), "status": "success"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing chat request: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.on_event("shutdown")
def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("Application shutting down")
    try:
        if 'client' in globals():
            client.close()
        logger.info("Cleanup completed successfully")
    except Exception as e:
        logger.error(f"Error during shutdown: {str(e)}")

# ========== 🔧 Error Handlers ==========

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    logger.error(f"HTTP Exception: {exc.status_code} - {exc.detail}")
    return {"error": exc.detail, "status_code": exc.status_code}

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    logger.error(f"Unhandled exception: {str(exc)}")
    logger.error(traceback.format_exc())
    return {"error": "Internal server error", "status_code": 500}

if __name__ == "__main__":
    import uvicorn
    logger.info("Starting FastAPI application")
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
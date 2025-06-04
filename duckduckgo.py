from bs4 import BeautifulSoup
import requests
import re
from urllib.parse import quote, unquote, urlparse, parse_qs

def scrape_url_content(url: str):
    print(f"Scraping content from: {url}")
    
    response = requests.get(url)
    
    if response.status_code == 200:
        # Parse the HTML of the page
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Extract the title
        title = soup.title.string if soup.title else "No Title"
        
        # Extract the meta description (if it exists)
        meta_desc = soup.find('meta', attrs={'name': 'description'})
        meta_desc_content = meta_desc['content'] if meta_desc else "No Meta Description"
        
        # Extract all <p> tags (for paragraphs)
        paragraphs = soup.find_all('p')
        
        # Extract all anchor tags <a> (for links)
        links = soup.find_all('a', href=True)
        
        # Extract all image URLs <img> tags
        images = soup.find_all('img', src=True)
        
        # Process the links: Extract URLs (only absolute URLs starting with http:// or https://)
        link_urls = [link['href'] for link in links if re.match(r'^https?://', link['href'])]
        
        # Process the images: Extract image sources (only absolute URLs starting with http:// or https://)
        image_urls = [img['src'] for img in images if re.match(r'^https?://', img['src'])]
        
        # Prepare content data (title, meta description, and first 3 paragraphs)
        content = {
            'title': title,
            'meta_description': meta_desc_content,
            'paragraphs': [p.get_text() for p in paragraphs[:3]]  # Get the first 3 paragraphs
        }
        
        # Return a dictionary with URLs, Image URLs, and Content
        result = {
            'urls': link_urls,
            'image_urls': image_urls,
            'content': content
        }
        
        return result
        
    else:
        print(f"Failed to retrieve the content from {url}. Status Code: {response.status_code}")
        return None


def extract_actual_urls(redirect_url: str):
    """Extract the actual URL from a DuckDuckGo redirect link."""
    parsed_url = urlparse(redirect_url)
    query_params = parse_qs(parsed_url.query)
    
    # Extract the 'uddg' parameter, which contains the actual URL
    if 'uddg' in query_params:
        actual_url = unquote(query_params['uddg'][0])  # Decode the URL
        return actual_url
    return None


def perform_web_search(query: str):
    print("Performing web search using DuckDuckGo (Unofficial API)...")
    # Convert spaces in the query to %20
    encoded_query = quote(query)
    
    # Construct the search URL with the encoded query
    search_url = f"https://html.duckduckgo.com/html?q={encoded_query}"
    
    # Make the request to DuckDuckGo
    response = requests.get(search_url)
    
    if response.status_code == 200:
        # Parse the HTML of the page
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Find the search result links
        results = []
        for link in soup.find_all('a', class_='result__a', href=True):
            redirect_url = link['href']
            actual_url = extract_actual_urls(redirect_url)  # Extract the real URL from the redirect
            if actual_url:
                results.append(actual_url)
        
        if results:
            return results
        else:
            print("No search result URLs found.")
            return None
    else:
        print(f"Failed to retrieve the search results. Status Code: {response.status_code}")
        return None


def main():
    # Perform the web search with a sample query.
    query = "credit card no cost emi offers in uk banks"
    results = perform_web_search(query)
    
    if results:
        print("Found URLs:")
        for url in results:
            print(url)
    else:
        print("No URLs found in the search results.")

# Run the main function
if __name__ == "__main__":
    main()

import os
import getpass
from langchain.tools import tool
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch 
from dotenv import load_dotenv
from bs4 import BeautifulSoup
import xml.etree.ElementTree as ET
import requests
import json

# For testing.
# os.chdir('c:\\Users\\justi\\Documents\\GitHub\\deploying-ai\\05_src\\')

# Instantiate chat client.
load_dotenv(".secrets")
chat_agent = ChatOpenAI(
    model="gpt-4o-mini",
    base_url="https://k7uffyg03f.execute-api.us-east-1.amazonaws.com/prod/openai/v1",
    api_key="any-value", 
    default_headers={
        "x-api-key": os.getenv("API_GATEWAY_KEY")
    }
)

# Do an API call to the search engine for a given search query.
@tool
def get_websearch(search_query:str) -> str:
    """
    An API call to Tavily is made.
    The API call is made to https://tavily.com/api/search.
    Accepted values for search_query are: Any string.
    """
    max_n = 5
    url_list = get_result_from_engine(search_query,max_n)
    out_webs = get_summary_from_result(search_query,url_list)
    return out_webs

# Get information from Tavily with the specified parameters based on what the API
# wants. Returns a list of URLs.
def get_result_from_engine(search_query:str,max_n:int):
    search = TavilySearch(api_key=os.getenv("TAVILY_API_KEY"),
                          max_results=max_n, 
                          description=f'tavily_search(query={search_query}) - a search engine.')
    results = search.run(search_query)  

    # Extract the URLs.
    url_list = []
    for result in results['results']:
        url_list.append(result['url'])
    return url_list

# Extract webpage text for a URL.
def extract_webpage_text(url: str) -> str:
    """
    Fetches a webpage and extracts visible text for summarization.
    """
    try:
        response = requests.get(url, timeout=5)
        response.raise_for_status()
    except Exception as e:
        print(f"Error fetching {url}: {e}")
        return ""
    soup = BeautifulSoup(response.text, "html.parser")
    for tag in soup(["script", "style", "header", "footer", "nav", "aside"]):
        tag.decompose()
    text = soup.get_text(separator="\n")
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return "\n".join(lines)

# From the returned results, summarize.
def get_summary_from_result(search_query:str,url_list:list) -> str:

    # For each URL.
    results = []
    for c_url in url_list:

        # Extract full web page text.
        page_text = extract_webpage_text(c_url)

        # Get LLM summary.
        summary_prompt = f"""
        Answer the search query based on the following web document from a given URL in 4 concise bullet points:
        
        Search Query:
        {search_query}

        URL:
        {c_url}

        Web Document: 
        {page_text}
        """
        web_summary = chat_agent.invoke(summary_prompt).content
        results.append(web_summary)
    
     # Join results with newlines and return.
    out_webs = '\n'.join(results)
    return out_webs

import os
from langchain.tools import tool
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
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

# Do an API call to arXiv to return preprints for a given search query.
@tool
def get_preprint(search_query:str) -> str:
    """
    An API call to arXiv is made.
    The API call is made to http://export.arxiv.org/api/query.
    Accepted values for search_query are: Any string.
    """
    response = get_preprint_from_service(search_query)
    pp_summary = get_summary_from_preprint(search_query,response)
    return pp_summary

# Get preprints from arXiv with the specified parameters based on what the API
# wants. Returns an XML response.
def get_preprint_from_service(search_query:str):
    url = 'http://export.arxiv.org/api/query'
    params = {
        'search_query': search_query
    }
    response = requests.get(url,params=params)
    return response

# From the returned preprints, summarize.
def get_summary_from_preprint(search_query:str, response:requests.Response) -> str:

    # Parse XML into separate entries.
    root = ET.fromstring(response.text)
    ns = {"atom": "http://www.w3.org/2005/Atom"}
    entries = root.findall("atom:entry",ns)
    if not entries:
        return 'No preprints found.'

    # For each preprint.
    results = []
    for entry in entries:

        # Extract information.
        title = entry.find("atom:title", ns).text.strip()
        abstract = entry.find("atom:summary", ns).text.strip()

        # Get LLM summary.
        summary_prompt = f"""
        Summarize the following research abstract in 4 concise bullet points:

        Title: {title}

        Abstract: 
        {abstract}
        """
        llm_summary = chat_agent.invoke(summary_prompt).content

        # Format information with LLM summary.
        formatted = f"""
        Title: {title}
        
        LLM Summary:
        {llm_summary}
        ----------------------------------------
        """
        results.append(formatted)
    
    # Join results with newlines and return.
    out_preprints = '\n'.join(results)
    return out_preprints

import requests
import xml.etree.ElementTree as ET
import chromadb
from langchain.tools import tool
from langchain_openai import ChatOpenAI
from chromadb.utils import embedding_functions
from dotenv import load_dotenv
from openai import OpenAI
import os

# For testing.
# os.chdir('c:\\Users\\justi\\Documents\\GitHub\\deploying-ai\\05_src\\')

# Instantiate OpenAI client and chat client.
load_dotenv(".secrets")
client = OpenAI(
    base_url="https://k7uffyg03f.execute-api.us-east-1.amazonaws.com/prod/openai/v1",
    api_key="any",
    default_headers={"x-api-key": os.getenv("API_GATEWAY_KEY")}
)
chat_agent = ChatOpenAI(
    model="gpt-4o-mini",
    base_url="https://k7uffyg03f.execute-api.us-east-1.amazonaws.com/prod/openai/v1",
    api_key="any-value", 
    default_headers={
        "x-api-key": os.getenv("API_GATEWAY_KEY")
    }
)

# Custom embedding function.
def custom_embed(texts):
    """
    texts: list[str]
    returns: list[list[float]] embeddings
    """
    if not texts:
        return []
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=texts
    )
    return [item.embedding for item in response.data]

# Open ChromaDB collection.
chroma = chromadb.HttpClient(host="localhost",port=8000)
collection = chroma.get_collection(name="pubmed_neuroscience")

# Query the collection with a semantic search.
@tool
def get_pubmed(search_query:str,n_results:int) -> str:
    """
    PubMed articles from the ChromaDB collection are retrieved for the search query.
    Accepted values for search_query are: Any string.
    """

    # Embed the search query and find relevant documents.
    embeddings = custom_embed([search_query])
    results = collection.query(
        query_embeddings=embeddings,
        n_results=n_results
    )
    contents = results["documents"][0]
    titles = results["metadatas"][0]
    nentries = len(contents)
    
    # For each preprint.
    results = []
    for entry in range(nentries):

        # Extract information.
        title = titles[entry]['title']
        abstract = contents[entry]

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
    out_pubmed = '\n'.join(results)
    return out_pubmed

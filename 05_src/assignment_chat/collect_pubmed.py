import requests
import xml.etree.ElementTree as ET
import chromadb
from chromadb.utils import embedding_functions
from dotenv import load_dotenv
from openai import OpenAI
import os

# Set number of results per query and each query to collect.
nresults = 10
queries = ['general intelligence and fMRI',
           'general intelligence theory']

# Instantiate OpenAI client.
load_dotenv(".secrets")
client = OpenAI(
    base_url="https://k7uffyg03f.execute-api.us-east-1.amazonaws.com/prod/openai/v1",
    api_key="any",
    default_headers={"x-api-key": os.getenv("API_GATEWAY_KEY")}
)

# Get PubMed results for specific query and number of results.
def fetch_pubmed(query: str,max_results:int):

    # Get PubMed IDs.
    search_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    search_params = {
        "db": "pubmed",
        "term": query,
        "retmax": max_results,
        "retmode": "json"
    }
    search_resp = requests.get(search_url, params=search_params)
    id_list = search_resp.json()["esearchresult"]["idlist"]
    if not id_list:
        return []

    # Fetch abstracts.
    fetch_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
    fetch_params = {
        "db": "pubmed",
        "id": ",".join(id_list),
        "retmode": "xml"
    }
    fetch_resp = requests.get(fetch_url, params=fetch_params)
    root = ET.fromstring(fetch_resp.text)
    papers = []
    for article in root.findall(".//PubmedArticle"):
        title_elem = article.find(".//ArticleTitle")
        abstract_elem = article.find(".//Abstract/AbstractText")
        title = title_elem.text if title_elem is not None else ""
        abstract = abstract_elem.text if abstract_elem is not None else ""
        if abstract:
            papers.append({
                "title": title,
                "abstract": abstract
            })
    return papers

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

# Start ChromaDB collection.
# openai_ef = embedding_functions.OpenAIEmbeddingFunction(
#     api_key=os.getenv("OPENAI_API_KEY"),
#     model_name="text-embedding-3-small"
# )
chroma = chromadb.HttpClient(host="localhost",port=8000)
collection = chroma.get_or_create_collection(
    name="pubmed_neuroscience",
    # embedding_function=openai_ef
)
# chroma.delete_collection(name="pubmed_neuroscience")

# Get the number of abstracts for each query.
for cquery in queries:

    # Fetch.
    papers = fetch_pubmed(cquery,nresults)
    documents = [p["abstract"] for p in papers]
    ids = [f"paper_{i}" for i in range(len(papers))]
    metadatas = [{"title": p["title"]} for p in papers]

    # Compute embeddings.
    embeddings = custom_embed(documents)

    # Add to collection, including embeddings.
    collection.add(
        documents=documents,
        ids=ids,
        metadatas=metadatas,
        embeddings=embeddings
    )

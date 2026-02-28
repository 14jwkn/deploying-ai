# Assignment 2: A Sample Response

This application implements three services. 

## Services

This implementation is based on LangGraph's tools. 

The file main.py contains the llm model calls that controls the chat. Tools are in the files tools_*.py.

### Service 1: API Calls

+ Implemented in tools_preprint_api.py
+ This tool does an API call to arXiv for a given search query, returns the relevant preprints, and summarizes each one.

### Service 2: Semantic Query

+ Documents for the ChromaDB collection were first collected, with embeddings generated, using collect_pubmed.py.
+ Implemented in tools_pubmed_rag.py.
+ This tool does a RAG by finding semantically similar documents to the search query in the collection and summarizing each one.

### Service 3: Web Search

+ Implemented in tools_websearch.py
+ This tool does a Tavily search for a given search query, returns the URLs, uses BeautifulSoup to parse the text in the URLs, and summarizes each one.

## User Interface

+ Added conversational style.
+ Implemented in Gradio.

---

## Guardrails and Other Limitations

* Include guardrails that prevent users from:

  * Accessing or revealing the system prompt.
  * Modifying the system prompt directly.

* The model must not respond to questions on certain restricted topics:

  * Cats or dogs
  * Horoscopes or Zodiac Signs
  * Taylor Swift

## Implementation

+ Spin up ChromaDB server by going to ./deploying_ai_data which contains a docker-compose.yml file and writing `docker compose up -d`, this will facilitate calculating distances for fetching.
+ Code is implemented in the folder `./05_src/assignment_chat`.

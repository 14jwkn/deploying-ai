from langgraph.graph import StateGraph, MessagesState, START
from langchain.chat_models import init_chat_model
from langgraph.prebuilt.tool_node import ToolNode, tools_condition
from langchain_core.messages import SystemMessage,  HumanMessage
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import json
import requests
import os
import xml.etree.ElementTree as ET
from utils.logger import get_logger
from assignment_chat.prompts import return_instructions
from assignment_chat.tools_preprint_api import get_preprint
from assignment_chat.tools_pubmed_rag import get_pubmed
from assignment_chat.tools_websearch import get_websearch

_logs = get_logger(__name__)
load_dotenv(".env")
load_dotenv(".secrets")

# Instantiate chat client, tools, and instruction prompt.
chat_agent = ChatOpenAI(
    model="gpt-4o-mini",
    base_url="https://k7uffyg03f.execute-api.us-east-1.amazonaws.com/prod/openai/v1",
    api_key="any-value", 
    default_headers={
        "x-api-key": os.getenv("API_GATEWAY_KEY")
    }
)
tools = [get_preprint,get_pubmed,get_websearch]
instructions = return_instructions()

# Bind tools to model and let them know if they should call a tool based on the instructions and messages.
# @traceable(run_type="llm")
def call_model(state: MessagesState):
    """LLM decides whether to call a tool or not"""
    response = chat_agent.bind_tools(tools).invoke([SystemMessage(content=instructions)] + state["messages"])
    return {
        "messages": [response]
    }

# Generate graphical work flow, attaching model node and tool nodes.
def get_graph():
    
    builder = StateGraph(MessagesState)
    builder.add_node(call_model)
    builder.add_node(ToolNode(tools))
    builder.add_edge(START, "call_model")
    builder.add_conditional_edges(
        "call_model",
        tools_condition,
    )
    builder.add_edge("tools", "call_model")
    graph = builder.compile()
    return graph

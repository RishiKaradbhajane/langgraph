from langgraph.graph import StateGraph, START, END
from langchain_groq import ChatGroq
from typing import TypedDict, Annotated, Literal
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
import operator
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage
import sqlite3

from langgraph.checkpoint.sqlite import SqliteSaver

load_dotenv()

model = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0
)


class ChatState(TypedDict):
    messages : Annotated[list[BaseMessage], add_messages]

def chat_node(state : ChatState):
    # Take user query from state
    messages = state['messages']
    # send to llm
    response = model.invoke(messages)
    # response store state
    return {'messages' : [response]}

conn = sqlite3.connect(database='chatbot.db', check_same_thread = False)
# Checkpointer
checkpointer = SqliteSaver(conn = conn)

graph = StateGraph(ChatState)

graph.add_node('chat_node', chat_node)
graph.add_edge(START, 'chat_node')
graph.add_edge('chat_node', END)

chatbot = graph.compile(checkpointer=checkpointer)

def retreive_all_threads():
    all_threads = set()
    for checkpoint in checkpointer.list(None):
        all_threads.add(checkpoint.config['configurable']['thread_id'])

    return list(all_threads)


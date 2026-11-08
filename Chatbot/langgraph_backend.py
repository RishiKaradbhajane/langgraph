from langgraph.graph import StateGraph, START, END
from langchain_groq import ChatGroq
from typing import TypedDict, Annotated, Literal
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
import operator
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage

from langgraph.checkpoint.memory import InMemorySaver

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

checkpointer = InMemorySaver()

graph = StateGraph(ChatState)

graph.add_node('chat_node', chat_node)
graph.add_edge(START, 'chat_node')
graph.add_edge('chat_node', END)

chatbot = graph.compile(checkpointer=checkpointer)

'''
thread_id = '1'

while True:
    user_message = input("Type here : ")

    print('User : ',user_message)

    if user_message.strip().lower() in ['exit', 'quit', 'bye','end'] :
        break

    config = {'configurable' : {'thread_id' : thread_id}}
    response = chatbot.invoke({'messages':[HumanMessage(content=user_message)]}, config = config)

    print('AI: ', response['messages'][-1].content)

'''



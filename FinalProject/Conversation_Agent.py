import os

from dotenv import load_dotenv, find_dotenv
from Models import fast_model
from langchain_core.messages import AIMessage, SystemMessage
from langgraph.graph import MessagesState, StateGraph, START, END

load_dotenv(find_dotenv())

if "OPENAI_API_KEY" not in os.environ:
    raise ValueError("OPENAI_API_KEY not found. Check your .env file.")


CONVERSATION_SYSTEM_MESSAGE = """You are the Conversation Agent in a multi-agent system.

Handle general questions and natural conversation.

Use the conversation history to understand the user's current message and
remember relevant details from earlier messages.

Respond naturally, helpfully, and concisely.

If the user's request is unclear or missing essential information, ask a
clarifying question. Do not say that you will route the request or recommend
another agent. Simply explain what information the user needs to provide.

Do not perform database queries, web research, document retrieval, or chart
creation. The Main Supervisor is responsible for routing those requests to
the appropriate specialized agents.

If the user asks what a specific named file or document says (e.g. a .pdf),
and no verified excerpt or summary from that file has been provided to you
in this conversation, do not answer from your own general knowledge of that
document, model, or paper -- even if you recognize the name. State plainly
that you do not have verified access to that document's contents in this
conversation and that it needs to be looked up, rather than describing what
it likely says.
"""

def conversation_node(state: MessagesState) -> dict:
    """Answer a general question using the conversation history."""

    if not state.get("messages"):
        return {"messages": [AIMessage(content="Please provide a question or message.")]}

    response = fast_model.invoke([SystemMessage(content=CONVERSATION_SYSTEM_MESSAGE)] + state["messages"])
    return {"messages": [response]}


conversation_builder = StateGraph(MessagesState)

conversation_builder.add_node("conversation", conversation_node)

conversation_builder.add_edge(START, "conversation")
conversation_builder.add_edge("conversation", END)

conversation_agent = conversation_builder.compile()
import chainlit as cl
from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage
from langchain_core.runnables import RunnableConfig
from Main_Supervisor import main_supervisor

USER_FACING_STREAM_NODES = {"conversation"}

def latest_ai_text(result: dict) -> str:
    """Return the latest complete user-facing AI response."""
    for message in reversed(result.get("messages", [])):
        if isinstance(message, AIMessage):
            return str(message.content)
    return "The system did not produce a response."


@cl.on_chat_start
async def on_chat_start():
    """Display the welcome message for a new Chainlit session."""
    await cl.Message(content=("Welcome! I can answer general questions, query the PostgreSQL "
                              "database, research the web, search internal documents, and "
                              "create bar, pie, or line charts.")).send()

@cl.on_message
async def on_message(message: cl.Message):
    """Stream a request through the Main Supervisor."""
    thread_id = cl.context.session.id
    callback = cl.LangchainCallbackHandler()
    config = RunnableConfig(configurable={"thread_id": thread_id},
                            recursion_limit=20,
                            callbacks=[callback])

    final_message = cl.Message(content="")
    final_state = None
    streamed_text = False

    try:
        stream = main_supervisor.astream({"messages": [HumanMessage(content=message.content)]},
                                         config=config,
                                         stream_mode=["messages", "values"])

        async for stream_mode, data in stream:
            if stream_mode == "messages":
                message_chunk, metadata = data
                node_name = metadata.get("langgraph_node")

                if (isinstance(message_chunk, AIMessageChunk)
                    and message_chunk.content
                    and node_name in USER_FACING_STREAM_NODES):
                    await final_message.stream_token(str(message_chunk.content))
                    streamed_text = True
                    
            elif stream_mode == "values":
                final_state = data

        if final_state is None:
            await cl.Message(content="The system did not return a final state.").send()
            return

        complete_answer = latest_ai_text(final_state)

        if not streamed_text:
            await final_message.stream_token(complete_answer)

        elements = []

        chart_config = final_state.get("chart_config")

        if chart_config:
            chart_element = cl.CustomElement(name="ChartViz",
                                             props={"config": chart_config},
                                             display="inline")
            elements.append(chart_element)
        final_message.elements = elements
        await final_message.send()

        metrics = (f"**Agents:** "
                   f"{', '.join(final_state.get('agents_called', [])) or 'None'}\n\n"
                   f"**Response time:** "
                   f"{final_state.get('total_response_time', 0):.2f} seconds\n\n"
                   f"**Retries:** "
                   f"{final_state.get('retry_count', 0)} · "
                   f"**Fallbacks:** "
                   f"{final_state.get('fallback_count', 0)}")

        await cl.Message(content=metrics,author="System Metrics").send()

    except Exception as error:
        await cl.Message(content=("The request could not be completed.\n\n"
                                  f"Error: {type(error).__name__}: {error}")).send()
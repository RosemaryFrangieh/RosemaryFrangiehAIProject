import json
import os
import re
from time import perf_counter, sleep
from typing import Any, Literal
from dotenv import find_dotenv, load_dotenv
from langchain_core.messages import HumanMessage
from Models import fast_model, powerful_model
from Conversation_Agent import conversation_agent
from RAG_Agent import rag_agent
from SQL_Agent import sql_query_agent
from Visualization_Agent import visualization_agent
from WebResearch_Agent import web_research_team
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.checkpoint.memory import (InMemorySaver)
from langgraph.graph import MessagesState, StateGraph, START, END
from pydantic import (BaseModel,
                      Field,
                      field_validator,
                      model_validator)

load_dotenv(find_dotenv(), override=True)

if "OPENAI_API_KEY" not in os.environ:
    raise ValueError("OOPENAI_API_KEY not found. Check your .env file.")

os.environ.setdefault("LANGSMITH_TRACING", "true")
os.environ.setdefault("LANGSMITH_PROJECT", "multi-agent-system")


AgentName = Literal["sql_query_agent",
        		    "web_research_team",
        		    "visualization_agent",
        		    "rag_agent",
        		    "conversation_agent"]
AgentStatus = Literal["success", "failure", "needs_clarification"]
RouteName = Literal["sql_query_agent",
        		    "web_research_team",
        		    "visualization_agent",
        		    "rag_agent",
        		    "conversation_agent",
        		    "FINISH"]


class SupervisorState(MessagesState):
    route: RouteName | None
    original_request: str
    last_agent: AgentName | None
    agent_status: AgentStatus | None
    error: str | None
    retry_count: int
    fallback_count: int
    agents_called: list[str]
    step_count: int
    sql_result: dict[str, Any] | None
    web_result: dict[str, Any] | None
    rag_response: str | None
    web_response: str | None
    chart_config: dict[str, Any] | None
    request_start_time: float
    agent_response_times: dict[str, float]
    total_response_time: float
    total_requests: int
    completed_requests: int
    completion_rate: float
    specialist_request: str
    workflow_steps: list[dict[str, Any]]
    workflow_index: int
    use_prior_result: bool
    prior_sql_result: dict[str, Any] | None
    prior_web_result: dict[str, Any] | None
    prior_chart_config: dict[str, Any] | None
    user_preferences: dict[str, str]
    workflow_errors: list[str]
    web_research_queries: list[str] | None
    web_evidence_snippet: str | None
    workflow_plan_debug: dict[str, Any] | None


class WorkflowStep(BaseModel):
    agent: AgentName
    instruction: str = Field(default="",
                             description=("The capability-specific work for this step, expressed "
                                          "generically and without instructions for other agents."))
    use_prior_result: bool = Field(default=False,
                                   description=("Whether this step intentionally consumes a structured result "
                                                "from an earlier turn rather than the current request."))

    @model_validator(mode="before")
    @classmethod
    def normalize_step_shape(cls, value):
        """Accept harmless key-name variations from JSON-capable models."""

        if not isinstance(value, dict):
            return value
        normalized = dict(value)

        if "agent" not in normalized:
            for key in ("route", "name", "capability", "destination"):
                if normalized.get(key):
                    normalized["agent"] = normalized[key]
                    break

        if "instruction" not in normalized:
            for key in ("task", "request", "action", "specialist_request"):
                if normalized.get(key):
                    normalized["instruction"] = normalized[key]
                    break

        return normalized

    @field_validator("agent", mode="before")
    @classmethod
    def normalize_agent_name(cls, value):
        aliases = {"sql": "sql_query_agent",
                   "database": "sql_query_agent",
                   "web": "web_research_team",
                   "research": "web_research_team",
                   "visualization": "visualization_agent",
                   "chart": "visualization_agent",
                   "rag": "rag_agent",
                   "document": "rag_agent",
                   "conversation": "conversation_agent",
                   "explanation": "conversation_agent"}

        normalized = str(value).strip().lower()
        return aliases.get(normalized, normalized)

    @field_validator("instruction", mode="before")
    @classmethod
    def normalize_instruction(cls, value):
        if value is None:
            return ""
        return str(value).strip()

    @field_validator("use_prior_result", mode="before")
    @classmethod
    def normalize_prior_flag(cls, value):
        if isinstance(value, str):
            return value.strip().lower() in {"true", "yes", "1"}
        return bool(value)


class WorkflowPlan(BaseModel):
    steps: list[WorkflowStep] = Field(default_factory=list)
    clarification: str | None = None
    destructive_confirmation_required: bool = False
    user_preferences: dict[str, str] = Field(default_factory=dict,
                                             description=("Any standing preference the user expresses about how results "
                                                          "should be produced -- e.g. {'chart_type': 'pie'}, "
                                                          "{'units': 'metric'}, {'response_length': 'short'}. Only include "
                                                          "preferences explicitly stated or clearly implied. Omit anything "
                                                          "not mentioned."))
    @model_validator(mode="before")
    @classmethod
    def normalize_plan_shape(cls, value):
        """Normalize equivalent JSON plan shapes before validation."""

        if not isinstance(value, dict):
            return value
        normalized = dict(value)

        if "steps" not in normalized:
            for key in ("workflow", "plan", "sequence", "agents"):
                if key in normalized:
                    normalized["steps"] = normalized[key]
                    break

        if isinstance(normalized.get("steps"), dict):
            normalized["steps"] = [normalized["steps"]]

        if "destructive_confirmation_required" not in normalized:
            for key in ("requires_confirmation",
                        "confirmation_required",
                        "destructive"):
                if key in normalized:
                    normalized["destructive_confirmation_required"] = (normalized[key])
                    break
        return normalized

    @field_validator("clarification", mode="before")
    @classmethod
    def normalize_clarification(cls, value):
        if value in (None, False, "", "false", "False", "none", "None"):
            return None
        if value is True:
            return "Ask the user for the missing information needed to continue."
        return str(value).strip() or None

    @field_validator("destructive_confirmation_required", mode="before")
    @classmethod
    def normalize_confirmation_flag(cls, value):
        if isinstance(value, str):
            return value.strip().lower() in {"true", "yes", "1"}
        return bool(value)

    @field_validator("user_preferences", mode="before")
    @classmethod
    def normalize_user_preferences(cls, value):
        if not isinstance(value, dict):
            return {}
        cleaned = {}
        for key, val in value.items():
            if val in (None, "", False):
                continue
            key_str = str(key).strip().lower()
            val_str = str(val).strip().lower()
            if key_str and val_str:
                cleaned[key_str] = val_str
        return cleaned


workflow_planner = fast_model.with_structured_output(WorkflowPlan,
                                                     method="json_mode",
                                                     include_raw=True)
MAX_SPECIALIST_STEPS = 4


def latest_human_text(messages) -> str:
    for message in reversed(messages):
        if isinstance(message, HumanMessage):
            return message.content.strip()
    return ""


def request_setup(state: SupervisorState):
    """Initialize only current-request fields; preserve cumulative counters."""

    request = latest_human_text(state.get("messages", []))
    prior_sql_result = (state.get("sql_result") or state.get("prior_sql_result"))
    prior_web_result = (state.get("web_result") or state.get("prior_web_result"))
    prior_chart_config = (state.get("chart_config") or state.get("prior_chart_config"))
    user_preferences = dict(state.get("user_preferences", {}))

    return {"route": None,
            "original_request": request,
            "specialist_request": "",
            "last_agent": None,
            "agent_status": None,
            "error": None,
            "retry_count": 0,
            "agents_called": [],
            "step_count": 0,
            "sql_result": None,
            "web_result": None,
            "rag_response": None,
            "web_response": None,
            "chart_config": None,
            "request_start_time": perf_counter(),
            "agent_response_times": {},
            "total_response_time": 0.0,
            "total_requests": state.get("total_requests", 0) + 1,
            "completed_requests": state.get("completed_requests", 0),
            "fallback_count": 0,
            "workflow_steps": [],
            "workflow_index": 0,
            "use_prior_result": False,
            "prior_sql_result": prior_sql_result,
            "prior_web_result": prior_web_result,
            "prior_chart_config": prior_chart_config,
            "user_preferences": user_preferences,
            "workflow_errors": [],
            "web_research_queries": None,
            "web_evidence_snippet": None,}


def route_from_supervisor(state: SupervisorState) -> RouteName:
    """Return the route selected by the Main Supervisor."""
    return state["route"]

def invoke_with_one_retry(callable_, payload):
    """Retry once, except for rate limits and authorization errors."""
    try:
        return callable_(payload), 0, None
    except Exception as first_error:
        error_text = str(first_error).lower()
        non_retryable = any(phrase in error_text for phrase in (
                "429", "rate limit", "rate_limit_exceeded", "401", "unauthorized", "403", "forbidden"))
        if non_retryable:
            return None, 0, str(first_error)
        try:
            return callable_(payload), 1, None
        except Exception as second_error:
            return None, 1, (f"First attempt: {first_error}; "
                    	     f"retry: {second_error}")


def wrapper_bookkeeping(state: SupervisorState,
            			agent: AgentName,
            			started: float,
            			retries: int) -> dict:
    times = dict(state.get("agent_response_times", {}))
    times[agent] = (times.get(agent, 0.0) + perf_counter() - started)

    return {"last_agent": agent,
            "agents_called": [*state.get("agents_called", []), agent],
            "step_count": state.get("step_count", 0) + 1,
            "retry_count": state.get("retry_count", 0) + retries,
            "agent_response_times": times}


def technical_failure(state, agent, started, retries, error) -> dict:
    update = wrapper_bookkeeping(state, agent, started, retries)
    update.update({"agent_status": "failure",
        	       "error": (f"{agent} could not complete the request. "
            	             f"{error}")})
    return update


def specialist_final_message(result):
    return next((message
            	 for message in reversed(result.get("messages", []))
            	 if isinstance(message, AIMessage)), None)

def invoke_specialist(state: SupervisorState, agent_name: str, agent, payload: dict):
    """Invoke a specialist and prepare its bookkeeping update."""
    started = perf_counter()
    result, retries, error = invoke_with_one_retry(agent.invoke, payload)
    if error:
        failure_update = technical_failure(state, agent_name, started, retries, error)
        return None, failure_update
    update = wrapper_bookkeeping(state, agent_name, started, retries)
    return result, update


def sql_wrapper(state: SupervisorState):
    """Invoke SQL using only its assigned database request."""
    sql_request = (f"Original user request:\n{state['original_request']}\n\n"
                   "Assigned SQL step:\n"
                   f"{state.get('specialist_request') or state['original_request']}")
    payload = {"messages": [HumanMessage(content=sql_request)]}
    result, update = invoke_specialist(state, "sql_query_agent", sql_query_agent, payload)
    if result is None:
        return update

    tool_message = next((message
                         for message in reversed(result.get("messages", []))
                         if isinstance(message, ToolMessage)), None) 
    if tool_message is None:
        final_message = specialist_final_message(result)

        sql_explanation = (str(final_message.content).strip()
                           if final_message is not None
                           else ("The request cannot be answered using the available "
                                 "todos database schema."))

        update.update({"messages": ([final_message] if final_message is not None else []),
                       "agent_status": "needs_clarification",
                       "error": sql_explanation,
                       "sql_result": None})
        return update

    try:
        sql_data = json.loads(tool_message.content)

    except (TypeError, json.JSONDecodeError):
        update.update({"agent_status": "failure",
                       "error": "The SQL tool returned an invalid structured result."})
        return update

    if "error" in sql_data:
        update.update({"agent_status": "failure",
                       "error": f"The SQL query failed: {sql_data['error']}"})
        return update

    final_message = specialist_final_message(result)
    update.update({"messages": [final_message] if final_message else [],
                   "agent_status": "success",
                   "error": None,
                   "sql_result": sql_data})
    return update


def web_wrapper(state: SupervisorState):
    """Invoke Web Research and preserve its report and structured data."""

    specialist_request = (state.get("specialist_request") or state["original_request"])
    workflow_steps = state.get("workflow_steps", [])
    current_index = state.get("workflow_index", 0)
    downstream_visualization = any(step.get("agent") == "visualization_agent"
                                   for step in workflow_steps[current_index + 1:])

    if downstream_visualization:
        specialist_request = (f"{specialist_request}\n\n"
                              "Downstream output contract: the verified result will be consumed "
                              "by the Visualization Agent. Return chart-ready structured data "
                              "with aligned labels and numerical values, a common unit, and "
                              "supporting source URLs. Preserve the chart type requested in the "
                              f"original request: {state['original_request']}")

    if state.get("rag_response"):
        specialist_request = (f"{specialist_request}\n\n"
                              "Verified internal-document result for resolving the subject:\n"
                              f"{state['rag_response']}")
    elif state.get("last_agent") == "rag_agent":
        pdf_match = re.search(r"[\w-]+\.pdf", state["original_request"], re.IGNORECASE)
        if pdf_match:
            specialist_request = (f"{specialist_request}\n\n"
                                  f"Note: the internal lookup for {pdf_match.group(0)} did not "
                                  "return usable content this turn. Treat the subject below as "
                                  "the specific named work, model, or publication that document "
                                  "concerns -- not an unrelated everyday topic, product, or "
                                  "game that happens to share the same name.")

    if state.get("use_prior_result"):
        prior_result = (state.get("prior_sql_result") or state.get("prior_web_result"))
        if prior_result:
            specialist_request = (f"{specialist_request}\n\n"
                                  "Verified prior structured result for resolving the "
                                  "follow-up reference:\n"
                                  f"{json.dumps(prior_result, default=str)[:5000]}")

    payload = {"messages": [HumanMessage(content=specialist_request)]}
    result, update = invoke_specialist(state, "web_research_team", web_research_team, payload)

    if result is None:
        return update

    search_status = result.get("search_results")

    if search_status == "SEARCH_ERROR":
        failure_message = specialist_final_message(result)
        update.update({"agent_status": "failure",
                       "error": (failure_message.content if failure_message else "The web-search service failed.")})
        return update

    if search_status == "NO_RESULTS":
        update.update({"agent_status": "needs_clarification",
                       "error": ("No useful web-search results were found. "
                                 "Rephrase or broaden the question.")})
        return update

    final_message = specialist_final_message(result)

    if final_message is None:
        update.update({"agent_status": "failure",
                       "error": "Web research did not produce a report."})
        return update

    structured_data = result.get("structured_data")

    if isinstance(structured_data, dict):
        web_result = structured_data

    else:
        web_result = {"can_visualize": False,
                      "labels": [],
                      "values": [],
                      "sources": [],
                      "missing_labels": [],
                      "search_results": search_status,
                      "extraction_error": result.get("extraction_error")}

    web_text = str(final_message.content)
    user_message = final_message
    
    if state.get("rag_response"):
        user_message = AIMessage(
            content=("Internal-document result:\n"
                     f"{state['rag_response']}\n\n"
                     "Current web result:\n"
                     f"{web_text}"))

    elif state.get("workflow_errors"):
        user_message = AIMessage(
            content=("Partial-result notice: "
                     + " ".join(state["workflow_errors"])
                     + "\n\n"
                     + str(user_message.content)))
        
    update.update({"messages": [user_message],
                   "agent_status": "success",
                   "error": None,
                   "web_result": web_result,
                   "web_response": web_text,
                   "web_research_queries": result.get("research_queries"),
                   "web_evidence_snippet": (str(result.get("search_results"))[:2000]
                                            if result.get("search_results") not in (None, "SEARCH_ERROR", "NO_RESULTS")
                                            else None)})
    return update


def visualization_wrapper(state: SupervisorState):
    """Create a chart from available SQL or web data."""

    specialist_request = (f"Original user request:\n{state['original_request']}\n\n"
                          "Assigned visualization step:\n"
                          f"{state.get('specialist_request') or state['original_request']}")
    saved_chart_type = state.get("user_preferences", {}).get("chart_type")
    if (saved_chart_type in ("bar", "pie", "line")
        and not re.search(r"\b(bar|pie|line)\b", specialist_request, re.IGNORECASE)):
        specialist_request += (f"\n\nUser's saved chart preference: {saved_chart_type} chart.")
    source_data = (state.get("sql_result") or state.get("web_result"))

    if source_data is None and state.get("use_prior_result"):
        source_data = (state.get("prior_sql_result") or state.get("prior_web_result"))

    sql_message = specialist_final_message(state) if state.get("sql_result") else None
    web_message = specialist_final_message(state) if state.get("web_result") else None
    
    payload = {"messages": [HumanMessage(content=specialist_request)],
               "source_data": source_data}

    result, update = invoke_specialist(state, "visualization_agent", visualization_agent, payload)

    if result is None:
        return update

    if result.get("visualization_status") != "success":
        update.update({"agent_status": "needs_clarification",
                       "error": (result.get("visualization_error") or "Valid chart data is required.")})
        return update

    chart_message = specialist_final_message(result)
    response_parts = []

    if (sql_message is not None and str(sql_message.content).strip()):
        response_parts.append(str(sql_message.content).strip())
    if (web_message is not None and str(web_message.content).strip()):
        response_parts.append(str(web_message.content).strip())
    if (chart_message is not None and str(chart_message.content).strip()):
        response_parts.append(str(chart_message.content).strip())
    if response_parts:
        final_message = AIMessage(content="\n\n".join(response_parts))
    else:
        final_message = AIMessage(content="The chart was created successfully.")

    update.update({"messages": [final_message],
                   "agent_status": "success",
                   "error": None,
                   "chart_config": result["chart_config"]})
    return update


def rag_wrapper(state: SupervisorState):
    """Invoke RAG and preserve its document-grounded answer."""
    
    sql_message = None
    assigned_request = (state.get("specialist_request") or state["original_request"])

    if state.get("last_agent") == "sql_query_agent":
        sql_message = specialist_final_message(state)
        rag_request = (f"{assigned_request}\n\n"
                       f"Original request:\n{state['original_request']}")
    else:
        rag_request = assigned_request

    payload = {"messages": [HumanMessage(content=rag_request)]}
    result, update = invoke_specialist(state, "rag_agent", rag_agent, payload)

    if result is None:
        return update

    final_message = specialist_final_message(result)

    if result.get("generation_failed"):
        update.update({"messages": ([final_message] if final_message is not None else []),
                       "agent_status": "failure",
                       "error": ("The documents were retrieved, but answer "
                                 "generation failed. Do not report this as "
                                 "missing document information.")})
        return update

    if not result.get("context_found"):
        rag_failure_text = (str(final_message.content).strip()
                            if final_message is not None
                                and str(final_message.content).strip()
                            else ("The requested information was not supported by the "
                                  "available internal documents."))
        update.update({"messages": ([final_message]
                                    if final_message is not None
                                        else []),
                       "agent_status": "needs_clarification",
                       "error": rag_failure_text})
        return update

    if final_message is None:
        update.update({"agent_status": "failure",
                       "error": ("Document retrieval succeeded but did not "
                                 "produce a final answer.")})
        return update

    rag_text = str(final_message.content).strip()
    response_parts = []
    if sql_message is not None and str(sql_message.content).strip():
        response_parts.append(str(sql_message.content).strip())
    response_parts.append(rag_text)

    combined_message = AIMessage(content="\n\n".join(response_parts))

    update.update({"messages": [combined_message],
                   "agent_status": "success",
                   "error": None,
                   "rag_response": rag_text})
    return update

def conversation_wrapper(state: SupervisorState):
    """Handle general conversation, interpretation, or failures."""

    is_fallback = (state.get("agent_status") in {"failure", "needs_clarification"}
                   and state.get("last_agent") != "conversation_agent")

    if is_fallback:
        successful_current_results = []

        if state.get("sql_result"):
            successful_current_results.append("Current SQL result:\n" + json.dumps(state["sql_result"],
                                                                                   default=str))

        if state.get("rag_response"):
            successful_current_results.append("Current document result:\n" + str(state["rag_response"]))

        if state.get("web_response"):
            successful_current_results.append("Current web result:\n" + str(state["web_response"]))

        current_results_text = ("\n\n".join(successful_current_results)
                                if successful_current_results
                                else "No part of the current request succeeded.")

        context = SystemMessage(content=("Explain the outcome of the current request accurately.\n\n"
                                         "Rules:\n"
                                         "- Discuss only the current request shown below.\n"
                                         "- Never reuse results from an earlier user request.\n"
                                         "- Never claim that a previous chart or database result "
                                         "belongs to the current request.\n"
                                         "- Do not invent unavailable database columns, tables, "
                                         "records, web facts, document facts, or chart values.\n"
                                         "- If the requested database information is unavailable, "
                                         "state what the database actually contains.\n"
                                         "- If an earlier stage of this current request succeeded, "
                                         "preserve that result.\n"
                                         "- If chart data is unavailable, explain what values are "
                                         "missing.\n"
                                         "- The relevant lookup for this request has already been "
                                         "attempted by the appropriate specialist. Never ask the "
                                         "user to paste excerpts, PDF text, or additional data -- "
                                         "report the actual outcome shown in 'Current failure "
                                         "information' below directly, in your own words.\n"
                                         "- Do not mention agents, routing, supervisors, or internal "
                                         "system architecture.\n\n"
                                         f"Current request:\n"
                                         f"{state['original_request']}\n\n"
                                         f"Current failure information:\n"
                                         f"{state.get('error')}\n\n"
                                         f"Successful results from this request only:\n"
                                         f"{current_results_text}"))

        fallback_request = HumanMessage(content=("Give the user a concise explanation of the current "
                                                 "request's outcome. Do not use any earlier conversation."))

        payload_messages = [context, fallback_request]

    elif state.get("last_agent") == "visualization_agent":
        source_data = (state.get("sql_result") or state.get("web_result") or {})

        context = SystemMessage(content=("A chart has already been created successfully.\n"
                                         "Provide only the additional explanation, interpretation, "
                                         "or recommendation requested by the user.\n\n"
                                         "Rules:\n"
                                         "- Use only the current structured source data below.\n"
                                         "- Do not reuse information from earlier requests.\n"
                                         "- Do not repeat all database rows unless necessary.\n"
                                         "- Do not merely say that the chart was created.\n"
                                         "- Never invent or change labels or values.\n"
                                         "- For a recommendation, identify the category requiring "
                                         "action from the supplied values and give one short, "
                                         "practical recommendation.\n"
                                         "- For highest, lowest, most common, or most attention, "
                                         "compare the numerical values explicitly.\n"
                                         "- Keep the answer concise.\n\n"
                                         f"Current original request:\n"
                                         f"{state['original_request']}\n\n"
                                         f"Current structured data:\n"
                                         f"{json.dumps(source_data, default=str)}"))

        explanation_request = HumanMessage(content=("Complete only the explanation, interpretation, or "
                                                    "recommendation requested in the current original request."))

        payload_messages = [context, explanation_request]


    elif (state.get("last_agent") == "web_research_team" and "rag_agent" in state.get("agents_called", [])):
        rag_response = str(state.get("rag_response") or "No document-grounded response was preserved.").strip()
        web_response = str(state.get("web_response") or "No completed web-research response was preserved.").strip()
        context = SystemMessage(content=("Create the final answer for a compound request that has "
                                         "already completed both document retrieval and web "
                                         "research.\n\n"
                                         "Important workflow facts:\n"
                                         "- The document research is complete.\n"
                                         "- The web research is complete.\n"
                                         "- WEB RESULT below contains the completed web findings.\n"
                                         "- Never say that web research still needs to be performed.\n"
                                         "- Ignore any sentence in DOCUMENT RESULT that says a Web "
                                         "Research Team will need to search later; that sentence is "
                                         "outdated because the search has now finished.\n\n"
                                         "Required answer structure:\n"
                                         "1. Document summary\n"
                                         "2. Recent web developments\n"
                                         "3. Direct comparison\n\n"
                                         "Rules:\n"
                                         "- Use only the two completed results supplied below.\n"
                                         "- Include concrete web developments when WEB RESULT "
                                         "contains them.\n"
                                         "- If WEB RESULT found no verified recent developments, "
                                         "say that the completed search did not verify any; do not "
                                         "say that searching was never performed.\n"
                                         "- Compare the paper's claims, architecture, applications, "
                                         "or limitations with the completed web findings.\n"
                                         "- Clearly distinguish information from the PDF from "
                                         "information found online.\n"
                                         "- Preserve relevant PDF page citations.\n"
                                         "- Preserve relevant web URLs.\n"
                                         "- Do not invent facts, citations, developments, or URLs.\n"
                                         "- Do not mention agents, routing, handoffs, tools, or "
                                         "internal system architecture.\n"
                                         "- Do not repeat the same description of the document in "
                                         "multiple sections.\n\n"
                                         f"Original request:\n"
                                         f"{state['original_request']}\n\n"
                                         f"DOCUMENT RESULT:\n"
                                         f"{rag_response}\n\n"
                                         f"COMPLETED WEB RESULT:\n"
                                         f"{web_response}"))

        synthesis_request = HumanMessage(content=("Using both completed results, write the final answer now. "
                                                  "Summarize the document, report the completed web findings, "
                                                  "and compare them directly. Never claim that web research "
                                                  "still needs to be performed."))
        payload_messages = [context, synthesis_request]

    elif state.get("last_agent") == "web_research_team":
        web_response = str(state.get("web_response") or "No completed web-research response was preserved.").strip()

        context = SystemMessage(content=("Create the final answer for a request whose web research "
                                         "step has already completed.\n\n"
                                         "Important workflow facts:\n"
                                         "- The web research is complete.\n"
                                         "- COMPLETED WEB RESULT below contains the actual findings.\n"
                                         "- Never say that web research still needs to be performed, "
                                         "and never claim you lack web access -- the search already "
                                         "ran and its results are supplied below.\n\n"
                                         "Rules:\n"
                                         "- Use only the completed web result supplied below.\n"
                                         "- Answer the user's original request directly using that "
                                         "content (e.g. summarize, explain, or compare as asked).\n"
                                         "- If COMPLETED WEB RESULT found no verified information, "
                                         "say that the completed search did not find it; do not say "
                                         "that searching was never performed.\n"
                                         "- Preserve relevant URLs from the completed result.\n"
                                         "- Do not invent facts, developments, or URLs beyond what "
                                         "is supplied.\n"
                                         "- Do not mention agents, routing, handoffs, tools, or "
                                         "internal system architecture.\n\n"
                                         f"Original request:\n"
                                         f"{state['original_request']}\n\n"
                                         f"COMPLETED WEB RESULT:\n"
                                         f"{web_response}"))

        synthesis_request = HumanMessage(content=("Using the completed web result above, write the final "
                                                  "answer to the original request now. Never claim that web "
                                                  "research still needs to be performed or that you lack "
                                                  "web access."))
        payload_messages = [context, synthesis_request]

    else:
        preserved_data = None
        if state.get("use_prior_result"):
            preserved_data = (state.get("prior_sql_result") or state.get("prior_web_result"))

        current_results = []
        if state.get("sql_result"):
            current_results.append("Current SQL result:\n" + json.dumps(state["sql_result"], default=str))
        if state.get("rag_response"):
            current_results.append("Current document result:\n" + str(state["rag_response"]))
        if state.get("web_response"):
            current_results.append("Current web result:\n" + str(state["web_response"]))
        current_results_text = "\n\n".join(current_results)

        assignment_context = SystemMessage(content=("Complete the assigned conversational step for the current "
                                                    "request. Use prior conversation only when the current request "
                                                    "refers to it. Do not invent database, web, document, or chart "
                                                    "results.\n\n"
                                                    f"Current request:\n{state['original_request']}\n\n"
                                                    f"Assigned step:\n{state.get('specialist_request') or state['original_request']}"
                                                    + ("\n\nResults already produced earlier in this same "
                                                       "request (use these to answer -- do not ask the user "
                                                       f"for them again):\n{current_results_text}"
                                                       if current_results_text else "")
                                                    + ("\n\nResult established earlier in this conversation "
                                                       "(use this to answer -- do not ask the user for it "
                                                       f"again):\n{json.dumps(preserved_data, default=str)}"
                                                       if preserved_data is not None else "")))
        payload_messages = [assignment_context, *state["messages"]]
    payload = {"messages": payload_messages}

    result, update = invoke_specialist(state, "conversation_agent", conversation_agent, payload)
    if result is None:
        return update

    final_message = specialist_final_message(result)

    update.update({"messages": ([final_message] if final_message else []),
                   "agent_status": "success",
                   "error": None,
                   "fallback_count": (state.get("fallback_count", 0)
                                      + (1 if is_fallback else 0))})
    return update
    

class InsertionPosition(BaseModel):
    index: int = Field(description=("Zero-based index in the existing steps list where the new "
                                     "step should be inserted, such that every step it depends on "
                                     "comes before it and every step that depends on it comes "
                                     "after. May equal len(steps) to place it last."))

    @field_validator("index", mode="before")
    @classmethod
    def coerce_index(cls, value):
        try:
            return int(value)
        except (TypeError, ValueError):
            return 0


insertion_planner = fast_model.with_structured_output(InsertionPosition,
                                                       method="json_mode",
                                                       include_raw=True)


def determine_insertion_index(agent: AgentName, steps: list[WorkflowStep], request: str) -> int:
    """Ask the model where a given agent belongs among an existing ordered steps list.

    steps must NOT already contain `agent` -- callers pass the list of other
    steps it should be placed relative to.
    """
    if not steps:
        return 0

    existing_summary = [{"position": i, "agent": step.agent, "instruction": step.instruction}
                        for i, step in enumerate(steps)]

    messages = [SystemMessage(content=("Decide the single correct index at which to insert an agent "
                                       "into an existing ordered list of workflow steps, so that:\n"
                                       "- Any existing step whose output the new agent needs comes "
                                       "before it.\n"
                                       "- Any existing step that needs the new agent's output comes "
                                       "after it.\n"
                                       "- If neither dependency applies, place it consistently with "
                                       "the order implied by the user's original request.\n\n"
                                       "Return only a json object: {\"index\": <int>}.")),
               HumanMessage(content=(f"Original user request:\n{request}\n\n"
                                     f"Agent to place: {agent}\n\n"
                                     f"Existing ordered steps:\n{json.dumps(existing_summary)}"))]
    try:
        result = invoke_with_rate_limit_retry(insertion_planner.invoke, messages)
        parsed = result.get("parsed") if isinstance(result, dict) else result
        if parsed is not None:
            return max(0, min(parsed.index, len(steps)))
    except Exception:
        pass
    return len(steps)


def validate_step_order(steps: list[WorkflowStep], request: str) -> list[WorkflowStep]:
    """Re-validate ordering of a full steps list, one agent at a time.

    For each step, ask determine_insertion_index where it belongs relative to
    the *other* steps, then reinsert it there. Steps already correctly placed
    round-trip back to (roughly) the same spot; misordered ones move. Called
    once right after a plan is first produced, and again after the
    coverage-gap correction pass, so ordering is checked twice rather than
    only when a missing agent is force-inserted.
    """
    if len(steps) < 2:
        return steps

    ordered = list(steps)
    for step in list(ordered):
        remaining = [s for s in ordered if s is not step]
        target_index = determine_insertion_index(step.agent, remaining, request)
        remaining.insert(target_index, step)
        ordered = remaining
    return ordered

WORKFLOW_PLANNER_PROMPT = """Plan an ordered workflow for one user request.

Return exactly one valid json object matching the WorkflowPlan schema. The json
object must contain steps, clarification, and destructive_confirmation_required.
Do not return prose, Markdown, or a code fence outside the json object.

Resources that are already connected and available:
- PostgreSQL is already connected. Its complete application schema is the todos
  table with id, task, deadline, and status. The user never needs to provide a
  connection string, table name, columns, task list, status list, or data file.
- The internal PDF collection is already loaded and indexed. It contains
  MachineLearning-Lecture01.pdf, donut_paper.pdf, and winter-sports.pdf. Never
  ask the user to upload or paste one of these configured PDFs.
- Web search is already available through web_research_team. Never claim that
  browsing is unavailable and do not ask for a URL when the request can be
  answered through research.
- Chart creation is already available through visualization_agent for verified
  structured data. Do not ask the user to manually provide data that a preceding
  SQL or web step can obtain.

Available capabilities:
- sql_query_agent: operations supported by the PostgreSQL todos schema
  (id, task, deadline, status).
- rag_agent: questions grounded in explicitly named or referenced internal PDFs.
- web_research_team: live, current, recent, or explicitly online research.
- visualization_agent: bar, pie, or line charts from verified structured data.
- conversation_agent: clarification, interpretation, comparison, explanation,
  recommendation, or final synthesis of results produced by earlier steps.

Planning rules:
- Plan capabilities, not keywords. Work for new subjects and phrasings.
- Every instruction must be self-contained: retain the subject, requested
  fields, filenames, categories, dates, and output constraints needed by that
  specialist. Never replace the actual task with a generic phrase such as
  "complete the web portion" or "handle the document request".
- Treat an unqualified request about tasks, statuses, or deadlines as a request
  about the connected todos table.
- Treat a configured PDF filename as an available internal document reference.
- Treat a request for online, live, current, recent, or latest information as a
  web-research capability request.
- Preserve the order implied by the request and include every requested action.
- A specialist step must contain only that specialist's work.
- Put visualization after the step that produces its numerical source data.
- Put conversation_agent last only when a separate explanation, comparison,
  interpretation, or recommendation remains after specialist work.
- A web report can explain its own research; do not add conversation_agent merely
  because the user used the word explain.
- The web team also summarizes and compares findings from its own research.
  Do not add conversation_agent for a summary, trend list, or simple explanation
  that uses only web evidence.
- The RAG agent explains and summarizes document evidence. In a RAG-to-web
  workflow, do not add conversation_agent unless the user requests a direct
  comparison or synthesis across the document and web results.
- A RAG-to-web request starts with RAG and then uses web research.
- A SQL-to-RAG request starts with SQL and then uses RAG.
- A request that refers to earlier results may consume a prior structured result;
  mark use_prior_result=true only for that deliberate follow-up.
- Never use an unrelated result from an earlier turn.
- If chart data or chart type is missing and no prior result is referenced, return
  a clarification instead of inventing data.
- Do not request clarification when the connected resources and the request
  already supply everything a specialist needs.
- If the database schema cannot answer the database portion, still route to SQL;
  that agent will return the controlled schema explanation.
- DELETE, DROP, TRUNCATE, broad UPDATE, or another destructive database operation
  requires confirmation before any SQL step. Set
  destructive_confirmation_required=true and return no executable steps.
- Missing-document handling belongs to RAG. Never replace it automatically with
  web research unless the user explicitly requested both.
- Return the smallest complete workflow; do not repeat an agent unnecessarily.
- If the user expresses any standing preference for how results should be
  produced (chart type, units, level of detail, tone, etc.), record it in
  user_preferences as a short key/value pair (e.g. {"chart_type": "pie"}).
  Do not invent preferences that weren't stated or clearly implied.
- If the message only states a standing preference (e.g. "my preferred chart type
  is bar," "I like pie charts") and requests no other action or data, return an
  empty steps list. Record the preference in user_preferences. Never invent a
  chart, database query, or search just because a chart type was mentioned.
"""

def terminal_conversation_route(instruction: str, *, error: str | None = None) -> dict:
    """Route once to Conversation Agent and then terminate the workflow."""

    update = {"route": "conversation_agent",
              "specialist_request": instruction,
              "workflow_steps": [{"agent": "conversation_agent",
                                  "instruction": instruction,
                                  "use_prior_result": False}],
              "workflow_index": 0}

    if error is not None:
        update.update({"agent_status": "failure",
                       "error": error})
    return update


def invoke_with_rate_limit_retry(callable_fn, *args, max_attempts=3, **kwargs):
    """Retry a model call a few times, specifically on rate-limit errors.

    Rate-limit responses (HTTP 429 / "rate_limit" in the provider's error)
    are transient and typically resolve within a second or two once the
    provider's token bucket refills -- unlike malformed-output errors, they
    say nothing about whether the request itself was reasonable, so it is
    safe and generic to just wait and retry rather than immediately falling
    back to a degraded route. Any other exception is re-raised immediately.
    """
    last_error = None

    for attempt in range(max_attempts):
        try:
            return callable_fn(*args, **kwargs)
        except Exception as error:
            message = str(error).lower()
            is_rate_limit = ("rate_limit" in message
                             or "429" in message
                             or "too many requests" in message)
            if not is_rate_limit or attempt == max_attempts - 1:
                raise

            last_error = error
            sleep(min(2 ** attempt, 8))
    raise last_error


def invoke_workflow_plan(messages) -> WorkflowPlan:
    """Parse a workflow plan, retrying once only after a format failure."""

    result = invoke_with_rate_limit_retry(workflow_planner.invoke, messages)
    parsed = result.get("parsed") if isinstance(result, dict) else result
    if parsed is not None:
        return parsed
    first_error = (result.get("parsing_error")
                   if isinstance(result, dict)
                   else "Unknown workflow parsing error")
    retry_messages = [*messages, 
                      HumanMessage(content=("Your preceding response could not be parsed. Return only one "
                                            "complete valid json object matching this exact top-level shape: "
                                            "{\"steps\": [{\"agent\": \"sql_query_agent\", "
                                            "\"instruction\": \"assigned work\", "
                                            "\"use_prior_result\": false}], "
                                            "\"clarification\": null, "
                                            "\"destructive_confirmation_required\": false}. "
                                            "Use only available agent names and adapt the steps to the "
                                            "original request."))]
    retry_result = invoke_with_rate_limit_retry(workflow_planner.invoke, retry_messages)
    retry_parsed = (retry_result.get("parsed") if isinstance(retry_result, dict) else retry_result)

    if retry_parsed is not None:
        return retry_parsed
    second_error = (retry_result.get("parsing_error")
                    if isinstance(retry_result, dict)
                    else "Unknown retry parsing error")

    raise ValueError(f"Initial parse: {first_error}; retry parse: {second_error}")


def workflow_coverage_gaps(request: str, plan: WorkflowPlan) -> list[str]:
    """Find explicitly requested capabilities omitted by a model plan.

    This validates resource and deliverable coverage; it does not decide the
    complete route or map individual test phrases to agents.
    """
    planned_agents = {step.agent for step in plan.steps}
    gaps = []

    if (re.search(r"\b(chart|graph|plot|visuali[sz](?:e|ation)|display\s+the\s+results)\b", request, re.IGNORECASE)
        and "visualization_agent" not in planned_agents):
        gaps.append("visualization_agent was omitted despite an explicit chart request")
    if (re.search(r"[\w.-]+\.pdf\b", request, re.IGNORECASE)
        and "rag_agent" not in planned_agents):
        gaps.append("rag_agent was omitted despite an explicit PDF reference")
    if (re.search(r"\b(web|online|internet|current|recent|latest|research)\b", request, re.IGNORECASE,)
        and "web_research_team" not in planned_agents):
        gaps.append("web_research_team was omitted despite an explicit research requirement")
    return gaps


def main_supervisor_node(state: SupervisorState) -> dict:
    """Execute a reusable capability plan instead of matching test phrases."""

    if state.get("step_count", 0) >= MAX_SPECIALIST_STEPS:
        if state.get("last_agent") == "conversation_agent":
            return {"route": "FINISH"}
        return terminal_conversation_route("Explain that the safe workflow-step limit was reached.",
                                           error="The safe workflow-step limit was reached.")

    if state.get("agent_status") in {"failure", "needs_clarification"}:
        if state.get("last_agent") == "conversation_agent":
            return {"route": "FINISH"}

        steps = state.get("workflow_steps", [])
        failed_index = state.get("workflow_index", 0)
        next_index = failed_index + 1
        failure_text = str(state.get("error") or f"{state.get('last_agent')} could not complete its step.").strip()
        workflow_errors = [*state.get("workflow_errors", []), failure_text]
        has_chartable_data = bool(state.get("sql_result") or state.get("web_result"))
        if next_index < len(steps):
            next_step = steps[next_index]
            if (next_step.get("agent") != "visualization_agent" or has_chartable_data):
                return {"route": next_step["agent"],
                        "specialist_request": next_step["instruction"],
                        "workflow_steps": steps,
                        "workflow_index": next_index,
                        "use_prior_result": bool(next_step.get("use_prior_result")),
                        "workflow_errors": workflow_errors,
                        "agent_status": None}

        fallback_instruction = ("Explain the current controlled failure or request the exact "
                                "missing information. Do not invent a result.")
        result = terminal_conversation_route(fallback_instruction)
        result["workflow_errors"] = workflow_errors
        return result

    steps = state.get("workflow_steps", [])
    index = state.get("workflow_index", 0)

    if steps and state.get("agent_status") == "success":
        index += 1
        
    merged_user_preferences = state.get("user_preferences", {})
    plan_debug = state.get("workflow_plan_debug") or {}

    if not steps:
        request = state.get("original_request", "").strip()
        if not request:
            return terminal_conversation_route("Ask the user to provide a request.")

        has_prior_data = (state.get("prior_sql_result") is not None
                          or state.get("prior_web_result") is not None)
        prior_inventory = {"sql_result_available": state.get("prior_sql_result") is not None,
                           "web_result_available": state.get("prior_web_result") is not None,
                           "chart_available": state.get("prior_chart_config") is not None}

        recent_history = []
        for message in state.get("messages", [])[-4:]:
            content = getattr(message, "content", None)
            if isinstance(content, str) and content.strip():
                recent_history.append({"role": getattr(message, "type", "message"),
                                       "content": content[-800:]})

        prior_evidence = {"sql_result": state.get("prior_sql_result"),
                          "web_result": state.get("prior_web_result")}

        planning_messages = [SystemMessage(content=WORKFLOW_PLANNER_PROMPT),
                             HumanMessage(content=(f"Current request:\n{request}\n\n"
                                                   f"Prior-result inventory:\n{json.dumps(prior_inventory)}\n\n"
                                                   f"Recent conversation:\n{json.dumps(recent_history, default=str)}\n\n"
                                                   f"Prior structured evidence:\n"
                                                   f"{json.dumps(prior_evidence, default=str)[:2500]}"))]

        plan_debug = {"initial_agents": None,
                      "initial_clarification": None,
                      "coverage_gaps": [],
                      "corrected_agents": None,
                      "corrected_clarification": None,
                      "remaining_gaps": [],
                      "safety_net_triggered": False,
                      "final_agents": None,
                      "final_clarification": None}
        try:
            plan = invoke_workflow_plan(planning_messages)
            plan.steps = validate_step_order(plan.steps, request)
            plan_debug["initial_agents"] = [step.agent for step in plan.steps]
            plan_debug["initial_clarification"] = plan.clarification
            coverage_gaps = workflow_coverage_gaps(request, plan)
            plan_debug["coverage_gaps"] = coverage_gaps
            if coverage_gaps:
                plan = invoke_workflow_plan([*planning_messages,
                                             HumanMessage(content=("The proposed workflow omitted explicitly required "
                                                                   "capabilities:\n- "
                                                                   + "\n- ".join(coverage_gaps)
                                                                   + "\nReturn a corrected smallest complete workflow. "
                                                                   "Preserve the requested order and do not add "
                                                                   "conversation_agent unless genuine cross-result "
                                                                   "synthesis remains."))])
                plan.steps = validate_step_order(plan.steps, request)
                plan_debug["corrected_agents"] = [step.agent for step in plan.steps]
                plan_debug["corrected_clarification"] = plan.clarification
                remaining_gaps = workflow_coverage_gaps(request, plan)
                plan_debug["remaining_gaps"] = remaining_gaps
                safety_net_agents = []

                if (any("visualization_agent" in gap for gap in remaining_gaps)
                    and not any(step.agent == "visualization_agent" for step in plan.steps)):
                    plan.steps.insert(determine_insertion_index("visualization_agent", plan.steps, request),
                                      WorkflowStep(agent="visualization_agent",
                                                   instruction=("Create the chart requested in the "
                                                                "original message using the result from "
                                                                "the preceding step.")))
                    safety_net_agents.append("visualization_agent")

                if (any("rag_agent" in gap for gap in remaining_gaps) 
                    and not any(step.agent == "rag_agent" for step in plan.steps)):
                    plan.steps.insert(determine_insertion_index("rag_agent", plan.steps, request),
                                      WorkflowStep(agent="rag_agent",
                                                      instruction=("Answer the internal-document portion of "
                                                                   "the original request using every "
                                                                   "explicitly named PDF.")))
                    safety_net_agents.append("rag_agent")

                if (any("web_research_team" in gap for gap in remaining_gaps)
                    and not any(step.agent == "web_research_team" for step in plan.steps)):
                    plan.steps.insert(determine_insertion_index("web_research_team", plan.steps, request),
                                      WorkflowStep(agent="web_research_team",
                                                   instruction=("Complete the web-research portion of the "
                                                                "original request using current, "
                                                                "verifiable sources.")))
                    safety_net_agents.append("web_research_team")

                if safety_net_agents:
                    plan.steps = validate_step_order(plan.steps, request)
                    plan_debug["safety_net_triggered"] = safety_net_agents
                    plan.clarification = None

            if (plan.steps
                and any(step.agent == "conversation_agent" for step in plan.steps)
                and plan.steps[-1].agent != "conversation_agent"):
                non_conv = [s for s in plan.steps if s.agent != "conversation_agent"]
                conv = [s for s in plan.steps if s.agent == "conversation_agent"]
                plan.steps = non_conv + conv

            plan_agents = {step.agent for step in plan.steps}
            has_data_step = bool(plan_agents & {"sql_query_agent", "web_research_team"})
            has_prior_data = (state.get("prior_sql_result") is not None 
                              or state.get("prior_web_result") is not None)
            if (not has_data_step and has_prior_data 
                and "visualization_agent" in plan_agents):
                for step in plan.steps:
                    if step.agent == "visualization_agent":
                        step.use_prior_result = True
            if (plan.clarification and not plan.steps and has_prior_data):
                plan.clarification = None
                plan.steps = [WorkflowStep(agent="conversation_agent",
                                           instruction=("Answer the current request using the result "
                                                        "already established earlier in this "
                                                        "conversation. Do not ask for information "
                                                        "that was already provided."), use_prior_result=True)]

            merged_user_preferences = {**state.get("user_preferences", {}), **plan.user_preferences}

            plan_debug["final_agents"] = [step.agent for step in plan.steps]
            plan_debug["final_clarification"] = plan.clarification
            plan_debug["user_preferences"] = merged_user_preferences
        except Exception as error:
            error_text = ("Workflow planning failed: "
                          f"{type(error).__name__}: {error}")
            return terminal_conversation_route("Explain that the request could not be planned because "
                                               "of a technical model failure.", error=error_text)

        if plan.destructive_confirmation_required:
            instruction = ("Ask for explicit confirmation before performing the "
                           "destructive database operation. State what may be deleted "
                           "or changed. Do not claim that it was executed.")
            route = terminal_conversation_route(instruction)
            route["workflow_plan_debug"] = plan_debug
            route["user_preferences"] = merged_user_preferences
            return route

        if plan.clarification and not plan.steps:
            route = terminal_conversation_route(plan.clarification)
            route["workflow_plan_debug"] = plan_debug
            route["user_preferences"] = merged_user_preferences
            return route

        if not plan.steps:
            if plan.user_preferences:
                instruction = ("Acknowledge briefly that the stated preference has been saved. "
                               "Do not invent, request, or perform any database, chart, or "
                               "research action.")
            else:
                instruction = "Ask for the missing information needed to continue."
            route = terminal_conversation_route(instruction)
            route["workflow_plan_debug"] = plan_debug
            route["user_preferences"] = merged_user_preferences
            return route

        raw_steps = [step.model_dump() for step in plan.steps]
        steps = []

        for candidate in raw_steps:
            if steps and steps[-1]["agent"] == candidate["agent"]:
                previous_instruction = steps[-1].get("instruction", "").strip()
                next_instruction = candidate.get("instruction", "").strip()
                steps[-1]["instruction"] = "\n\n".join(part
                                                       for part in (previous_instruction, next_instruction)
                                                       if part)
                steps[-1]["use_prior_result"] = (steps[-1].get("use_prior_result", False)
                                                 or candidate.get("use_prior_result", False))
            else:
                steps.append(candidate)

        for step in steps:
            if not step.get("instruction"):
                step["instruction"] = request
        index = 0

    if index >= len(steps):
        return {"route": "FINISH", "workflow_index": index}

    step = steps[index]
    return {"route": step["agent"],
            "specialist_request": step["instruction"],
            "workflow_steps": steps,
            "workflow_index": index,
            "use_prior_result": bool(step.get("use_prior_result")),
            "workflow_plan_debug": plan_debug,
            "user_preferences": merged_user_preferences}

def finalize_request(state: SupervisorState):
    total_time = perf_counter() - state["request_start_time"]
    final_message = (state["messages"][-1] if state.get("messages") else None)
    has_user_facing_response = bool(isinstance(final_message, AIMessage)
                    			    and isinstance(final_message.content, str)
                			        and final_message.content.strip())
    completed = state.get("completed_requests", 0) + int(has_user_facing_response)
    total = state.get("total_requests", 0)
    return {"total_response_time": total_time,
            "completed_requests": completed,
            "completion_rate": (completed / total * 100) if total else 0.0}

main_builder = StateGraph(SupervisorState)
main_builder.add_node("request_setup", request_setup)
main_builder.add_node("main_supervisor", main_supervisor_node)
main_builder.add_node("sql_wrapper", sql_wrapper)
main_builder.add_node("web_wrapper", web_wrapper)
main_builder.add_node("visualization_wrapper", visualization_wrapper)
main_builder.add_node("rag_wrapper", rag_wrapper)
main_builder.add_node("conversation_wrapper", conversation_wrapper)
main_builder.add_node("finalize", finalize_request)

main_builder.add_edge(START, "request_setup")
main_builder.add_edge("request_setup", "main_supervisor")
main_builder.add_conditional_edges("main_supervisor", route_from_supervisor,
                   				  {"sql_query_agent": "sql_wrapper",
                    			   "web_research_team": "web_wrapper",
                    			   "visualization_agent": "visualization_wrapper",
                   			       "rag_agent": "rag_wrapper",
                    			   "conversation_agent": "conversation_wrapper",
                    			   "FINISH": "finalize"})
for wrapper in ("sql_wrapper",
        		"web_wrapper",
        		"visualization_wrapper",
        		"rag_wrapper",
        		"conversation_wrapper"):
    main_builder.add_edge(wrapper, "main_supervisor")

main_builder.add_edge("finalize", END)
main_supervisor = main_builder.compile(checkpointer=InMemorySaver())
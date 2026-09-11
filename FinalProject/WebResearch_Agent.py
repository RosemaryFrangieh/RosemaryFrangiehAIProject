
import os
import re

from typing import Any, Literal
from datetime import date
from urllib.parse import urlsplit, urlunsplit
from pydantic import BaseModel, Field, model_validator
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.graph import MessagesState, StateGraph, START, END
from Models import fast_model, powerful_model, secondary_model

if "TAVILY_API_KEY" not in os.environ:
    raise ValueError("TAVILY_API_KEY not found. Check your .env file.")


class SearchPlan(BaseModel):
    """Targeted searches needed to answer one web request."""
    queries: list[str] = Field(description=("Two to six focused web-search queries. For charts and "
                                            "comparisons, search separately for requested years, "
                                            "countries, categories, or missing data points."))
class ExtractedWebDataset(BaseModel):
    """Verified numerical data extracted from web evidence."""
    can_visualize: bool = Field(description=("True only when enough explicit numerical values exist "
                                             "to create the requested chart."))

    chart_type: Literal["bar", "pie", "line"] | None = Field(default=None,
                                                             description="The chart type requested by the user.")
    title: str | None = Field(default=None, description="A short title for the dataset.")
    labels: list[str] = Field(default_factory=list, description="The labels explicitly supported by the evidence.")
    values: list[Any] = Field(default_factory=list, description=("The numerical values explicitly stated in the evidence."))
    unit: str | None = Field(default=None, description=("The common measurement unit, such as percent or people."))
    sources: list[str] = Field(default_factory=list, description=("URLs from the supplied evidence that support the values."))
    missing_labels: list[str] = Field(default_factory=list, description=("Requested labels whose values could not be verified."))
    explanation: str | None = Field(default=None, description=("Why the dataset can or cannot be visualized."))

    @model_validator(mode="before")
    @classmethod
    def normalize_dataset_shape(cls, value):
        """Keep imperfect extraction output available for safe recovery."""

        if isinstance(value, dict):
            normalized = dict(value)
            if "sources" not in normalized and "source_urls" in normalized:
                normalized["sources"] = normalized["source_urls"]
            return normalized
        return value

search_planner = secondary_model.with_structured_output(SearchPlan, method="json_mode")
dataset_extractor = powerful_model.with_structured_output(ExtractedWebDataset, method="json_mode")


class RequestedDateScope(BaseModel):
    """The literal set of data points the user asked for, if any."""
    has_explicit_scope: bool = Field(description=("True only if the user stated or implied a literal, "
                                                   "countable span or list of points -- e.g. an explicit "
                                                   "year, a year range, 'last N years', a century, a "
                                                   "decade, a list of named countries/products, etc. "
                                                   "False if the request has no literal countable scope."))
    unit: str | None = Field(default=None, description=("The unit of one point, e.g. 'year', 'month', "
                                                         "'decade', 'century', 'country', 'product'. "
                                                         "Only set when has_explicit_scope is true."))
    point_labels: list[str] = Field(default_factory=list,
                                    description=("The literal, ordered labels for each point -- computed "
                                                 "in full from any range the user gave (e.g. '2010-2013' "
                                                 "becomes ['2010','2011','2012','2013'], 'the 1500s' "
                                                 "becomes ['1500'..'1599'] only if truly needed, 'last "
                                                 "11 years' becomes the 11 most recent calendar years "
                                                 "counting back from today). Never invent labels the user "
                                                 "did not imply."))

    @property
    def point_count(self) -> int | None:
        return len(self.point_labels) if self.has_explicit_scope and self.point_labels else None


date_scope_extractor = secondary_model.with_structured_output(RequestedDateScope, method="json_mode")


def requested_date_scope(request: str) -> RequestedDateScope:
    """Ask the model to identify any literal, countable scope in the request."""
    today = date.today().isoformat()
    prompt = SystemMessage(content=("Return one valid json object identifying whether the user's request "
                                    "names a literal, countable set of data points, and if so, what they "
                                    "are.\n\n"
                                    f"Today's date is {today}.\n\n"
                                    "Rules:\n"
                                    "- Only set has_explicit_scope to true for a literal span or list: an "
                                    "explicit year, an explicit year/date range, 'last/past N "
                                    "years/months/decades', an explicit century or decade, or an explicit "
                                    "list of named items (countries, products, categories).\n"
                                    "- If a range or 'last N units' is given, expand it into the full "
                                    "ordered list of point_labels yourself, computed correctly from "
                                    "today's date where relevant. Do not just restate the range.\n"
                                    "- If the request has no literal countable scope (e.g. 'show me GDP "
                                    "growth' with no range or count), set has_explicit_scope to false and "
                                    "leave point_labels empty.\n"
                                    "- Never invent labels beyond what the user's wording supports.\n"))
    try:
        return date_scope_extractor.invoke([prompt, HumanMessage(content=request)])
    except Exception:
        return RequestedDateScope(has_explicit_scope=False)


class WebResearchState(MessagesState):
    route: str
    search_results: str
    research_queries: list[str]
    research_round: int
    extraction_complete: bool
    structured_data: dict | None
    extraction_error: str | None


def latest_request(state: WebResearchState) -> str:
    """Return the latest user request."""
    for message in reversed(state.get("messages", [])):
        if isinstance(message, HumanMessage):
            return message.content.strip()
    return ""

def research_subject(request: str) -> str:
    """Remove supervisor handoff metadata from the search subject."""
    return request.split("\n\nDownstream output contract:", 1)[0].strip()

def chart_requested(request: str) -> bool:
    """Return whether the request asks for a chart."""
    return bool(re.search(r"\b(chart|graph|visualize|visualise|plot|bar|pie|line)\b",
                          request, re.IGNORECASE))

def requested_chart_type(request: str) -> str | None:
    """Return the explicitly requested chart type."""
    match = re.search(r"\b(bar|pie|line)\b", request, re.IGNORECASE)
    if match:
        return match.group(1).lower()
    return None

def requested_years(request: str) -> list[str]:
    """Return explicitly requested years: 1000s/2000s-style years, or any year tagged with an era."""
    plain_years = re.findall(r"\b[12]\d{3}\b", request)

    tagged_years = [f"{number} {era.upper()}"
                    for number, era in re.findall(r"\b(\d{1,4})\s*(AD|CE|BC|BCE)\b", request, re.IGNORECASE)]

    return list(dict.fromkeys([*plain_years, *tagged_years]))

def required_point_count(request: str) -> int:
    """Return the minimum number of expected chart points."""
    scope = requested_date_scope(request)
    return scope.point_count or 2


def valid_source_urls(search_results: str) -> set[str]:
    """Return URLs that genuinely appeared in search evidence."""
    return set(re.findall(r'<Document href="([^"]+)">', search_results))

def canonical_source_url(url: str) -> str:
    """Normalize harmless URL variations without changing source identity."""
    try:
        parts = urlsplit(str(url).strip())
        host = parts.netloc.lower()
        if host.startswith("www."):
            host = host[4:]
        path = parts.path.rstrip("/") or "/"
        return urlunsplit((parts.scheme.lower(), host, path, "", ""))
    except (TypeError, ValueError):
        return ""


def verified_numeric_value(value: Any) -> float | None:
    """Parse one complete numeric value; reject malformed concatenations."""
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        text = value.strip()
        match = re.match(r"[-+]?\d{1,3}(?:,\d{3})*(?:\.\d+)?%?", text)
        if match and match.group(0):
            return float(match.group(0).rstrip("%").replace(",", ""))
    return None


def sub_supervisor(state: WebResearchState) -> dict:
    """Coordinate searching, extraction, refinement, and reporting."""
    request = latest_request(state)
    if not request:
        return {"route": "end",
                "messages": [AIMessage(content="Please provide a research question.")]}

    search_results = state.get("search_results")

    if not search_results:
        return {"route": "researcher"}
    if search_results == "SEARCH_ERROR":
        return {"route": "end"}
    if search_results == "NO_RESULTS":
        return {"route": "end", 
                "messages": [AIMessage(content=("No useful web-search results were found."))]}

    extraction_complete = state.get("extraction_complete", False)
    if chart_requested(request):
        if not extraction_complete:
            return {"route": "data_extractor"}
            
        structured_data = (state.get("structured_data") or {})
        
        if structured_data.get("can_visualize") is True:
            return {"route": "report_writer"}

        missing_labels = structured_data.get("missing_labels", [])
        research_round = state.get("research_round", 0)
        if missing_labels and research_round < 4:
            return {"route": "researcher"}

    elif not extraction_complete and re.search(r"\d", search_results):
        return {"route": "data_extractor"}
    return {"route": "report_writer"}
    

def route_request(state: WebResearchState) -> Literal["researcher",
                                                      "data_extractor",
                                                      "report_writer",
                                                      "end"]:
    """Return the route selected by the sub-supervisor."""
    return state["route"]


def plan_search_queries(request: str,
                        missing_labels: list[str] | None = None,
                        already_tried: list[str] | None = None) -> list[str]:
    """Create generic targeted searches for requested evidence."""
    request = research_subject(request) or request
    missing_labels = missing_labels or []
    already_tried = already_tried or []
    
    if missing_labels:
        clean_request = re.sub(r"\b(?:then\s+)?(?:create|make|display|show|present)\b.*$",
                               "",
                               request,
                               flags=re.IGNORECASE).strip(" ,.") or request

        def static_templates() -> list[str]:
            templates = []
            for label in missing_labels:
                templates.extend([f'"{label}" {clean_request} exact value and unit',
                                  f'"{label}" {clean_request} statistics authoritative source'])
            return list(dict.fromkeys(templates))

        retry_prompt = SystemMessage(content=("Return one valid json object containing focused web-search "
                                              "queries to find values for labels that previous searches "
                                              "could not verify.\n\n"
                                              "Rules:\n"
                                              "- The following labels still lack a verified numerical "
                                              f"value: {', '.join(missing_labels)}.\n"
                                              "- The queries below were already tried and did not "
                                              "produce a verified value. Do not repeat any of them "
                                              "verbatim or with only trivial wording changes -- vary "
                                              "the general phrasing or angle instead:\n"
                                              + "\n".join(f"  - {q}" for q in already_tried[-12:])
                                              + "\n"
                                              "- Do not target a specific named website, publisher, or "
                                              "'site:' filter unless you are confident it currently "
                                              "publishes this data as readable text (not a paywalled "
                                              "or chart-only page). When unsure, use general phrasing "
                                              "instead of naming a source.\n"
                                              "- Never invent or guess the name of a data provider, "
                                              "market-research firm, or website.\n"
                                              "- Generate one or two new queries per missing label.\n"
                                              "- Never invent facts or numerical values.\n"
                                              "- Return between two and six focused queries.\n"))
        try:
            plan = search_planner.invoke([retry_prompt, HumanMessage(content=clean_request)])
            queries = [query.strip()
                       for query in plan.queries
                           if isinstance(query, str) and query.strip()]
            queries = [query for query in queries if query not in already_tried]
        except Exception:
            queries = []

        if not queries:
            queries = [query for query in static_templates() if query not in already_tried]

        if not queries:
            queries = static_templates()

        return queries[:6]

    planning_prompt = SystemMessage(content=("Return one valid json object containing focused web-search "
                                             "queries that will answer the "
                                             "user's research request.\n\n"
                                             "Rules:\n"
                                             "- Identify the actual subject and requested information.\n"
                                             "- Use supplied context to resolve pronouns and ambiguous "
                                             "references.\n"
                                             "- Exclude unrelated subjects that merely share a name.\n"
                                             "- For numerical or chart requests, identify every requested "
                                             "label and search for explicit values and units.\n"
                                             "- For current or recent requests, include recency terms.\n"
                                             "- Prefer one authoritative source covering the full request.\n"
                                             "- Create separate targeted queries when different labels or "
                                             "subquestions require different evidence.\n"
                                             "- Remove instructions about writing, summarizing, explaining, "
                                             "or displaying a chart from the search wording.\n"
                                             "- Prefer primary, official, or authoritative sources.\n"
                                             "- Return between two and six focused queries.\n"
                                             "- Never invent facts or numerical values.\n"))

    try:
        plan = search_planner.invoke([planning_prompt, HumanMessage(content=request)])
        queries = [query.strip() for query in plan.queries if isinstance(query, str) and query.strip()]

    except Exception:
        queries = []

    queries = list(dict.fromkeys(queries))
    
    clean_request = re.sub(r"\b(?:then\s+)?(?:create|make|display|show|present)\s+"
                           r"(?:the\s+results\s+as\s+)?(?:a\s+|an\s+)?"
                           r"(?:bar|pie|line)?\s*(?:chart|graph|plot)\b.*$",
                           "",
                           request,
                           flags=re.IGNORECASE).strip(" ,.")

    if clean_request and clean_request not in queries:
        queries.insert(0, f"{clean_request} exact numerical values authoritative source")

    if not queries:
        queries = [f"{request} exact numerical values authoritative source"]

    return queries[:6]


def researcher(state: WebResearchState) -> dict:
    """Perform one or more targeted web searches."""

    request = latest_request(state)

    if not request:
        return {"search_results": "NO_RESULTS",
                "research_queries": []}

    structured_data = (state.get("structured_data") or {})
    missing_labels = structured_data.get("missing_labels", [])
    
    queries = plan_search_queries(request,
                                  missing_labels=missing_labels,
                                  already_tried=state.get("research_queries", []))

    previous_results = state.get("search_results")

    if previous_results in {None, "", "NO_RESULTS", "SEARCH_ERROR"}:
        previous_results = ""

    collected_documents = []
    seen_documents = set()
    errors = []
    
    for query in queries:
        try:
            import json
            import subprocess

            payload = json.dumps({
                "api_key": os.environ["TAVILY_API_KEY"],
                "query": query,
                "max_results": 7,
                "search_depth": "advanced",
            })

            result = subprocess.run(
                ["curl", "-s", "https://api.tavily.com/search",
                 "-X", "POST",
                 "-H", "Content-Type: application/json",
                 "-d", payload],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=30,
            )

            if result.returncode != 0:
                raise RuntimeError(f"curl error: {result.stderr}")

            response = json.loads(result.stdout)
            search_docs = response.get("results", [])

        except Exception as error:
            errors.append(f"{type(error).__name__}: {error}")
            continue

        for document in search_docs:
            if not isinstance(document, dict):
                continue

            url = document.get("url")
            body = document.get("content")

            if not url or not body:
                continue

            document_key = (url, body.strip())
            if document_key in seen_documents:
                continue

            seen_documents.add(document_key)
            collected_documents.append({"query": query,
                                        "title": document.get("title", "Untitled"),
                                        "href": url,
                                        "body": re.sub(r"\s+", " ", body).strip()[:1200]})
            if len(collected_documents) >= 18:
                break
        if len(collected_documents) >= 18:
            break

    if not collected_documents:
        if errors:
            return {"search_results": "SEARCH_ERROR",
                    "research_queries": queries,
                    "messages": [AIMessage(content=("Web search failed: " + "; ".join(errors)))]}

        return {"search_results": "NO_RESULTS",
                "research_queries": queries}

    formatted_results = "\n\n---\n\n".join((f'<Document href="{document["href"]}">\n'
                                            f'Search query: {document["query"]}\n'
                                            f'Title: {document["title"]}\n'
                                            f'{document["body"]}\n'
                                            f"</Document>")
                                           for document in collected_documents)[:16000]

    combined_results = formatted_results

    if previous_results:
        combined_results = (previous_results[-8000:]
                            + "\n\n--- FOLLOW-UP SEARCH ---\n\n"
                            + formatted_results[:10000])

    return {"search_results": combined_results,
            "research_queries": [*state.get("research_queries", []), *queries],
            "research_round": (state.get("research_round", 0) + 1),
            "extraction_complete": False,
            "structured_data": None,
            "extraction_error": None}


def extract_structured_data(state: WebResearchState) -> dict:
    """Extract and validate chart-ready web data."""

    request = latest_request(state)
    subject = research_subject(request) or request
    search_results = state["search_results"]

    extraction_prompt = SystemMessage(content=("Return one valid json object containing one chart-ready "
                                               "numerical dataset using only "
                                               "the supplied web-search evidence.\n\n"
                                               "Strict rules:\n"
                                               "- Use only values explicitly written in the evidence.\n"
                                               "- Never estimate, interpolate, extrapolate, or invent.\n"
                                               "- Do not turn one overall percentage change into "
                                               "individual annual values.\n"
                                               "- Keep all values on the same metric and unit.\n"
                                               "- Do not mix TIOBE, PYPL, GitHub, Stack Overflow, or "
                                               "other measurements into one dataset.\n"
                                               "- Every source URL must appear exactly in the evidence.\n"
                                               "- Labels and values must have equal lengths.\n"
                                               "- Preserve chronological order for a time series.\n"
                                               "- First identify every label explicitly requested by the "
                                               "user, regardless of whether the labels represent years, "
                                               "countries, products, categories, technologies, or anything "
                                               "else.\n"
                                               "- Produce exactly one numerical value for every requested "
                                               "label.\n"
                                               "- Preserve the user's label names in the labels field.\n"
                                               "- For non-time-series comparisons (e.g. countries, products, "
                                               "categories), each label's value may come from that label's "
                                               "own most recently reported figure in the evidence -- labels "
                                               "do not need to share the same reference year as each other. "
                                               "Do not require or invent a specific year for a label unless "
                                               "the user explicitly asked for that exact year; use the most "
                                               "recent verified figure available for that label instead.\n"
                                               "- If any requested label lacks an explicit verified value, "
                                               "set can_visualize to false and put that exact label in "
                                               "missing_labels.\n"
                                               "- Preserve any verified labels and values even when the "
                                               "dataset is incomplete, so targeted follow-up research can "
                                               "identify what remains missing.\n"
                                               "- If the dataset is incomplete, preserve verified partial "
                                               "labels and values, list every missing label, and never fill "
                                               "a missing value by estimation.\n\n"
                                               f"Web-search evidence:\n"
                                               f"{search_results}"))

    try:
        extracted = dataset_extractor.invoke([extraction_prompt, HumanMessage(content=subject)])

    except Exception as first_error:
        try:
            extracted = dataset_extractor.invoke([extraction_prompt,
                                                  HumanMessage(content=subject),
                                                  AIMessage(content=("My previous response could not be parsed: "
                                                                     f"{first_error}")),
                                                  HumanMessage(content=("Return exactly one JSON object (never a list) "
                                                                        "with these top-level keys: can_visualize, "
                                                                        "chart_type, title, labels, values, unit, "
                                                                        "sources, missing_labels, explanation. Use the "
                                                                        "same underlying values you already identified."))])

        except Exception as second_error:
            return {"extraction_complete": True,
                    "structured_data": None,
                    "extraction_error": ("The web evidence could not be converted into "
                                         "structured data after 2 attempts: "
                                         f"{type(second_error).__name__}: {second_error}")}
    data = extracted.model_dump()

    raw_labels = [str(label).strip()
                  for label in data.get("labels", [])]
    raw_values = data.get("values", [])
    labels = []
    values = []
    invalid_value_labels = []

    for index, label in enumerate(raw_labels):
        raw_value = (raw_values[index] if index < len(raw_values) else None)
        numeric_value = verified_numeric_value(raw_value)
        if numeric_value is None:
            invalid_value_labels.append(label)
        else:
            labels.append(label)
            values.append(numeric_value)

    sources = data.get("sources", [])

    available_urls = valid_source_urls(search_results)

    evidence_url_by_canonical = {canonical_source_url(url): url
                                 for url in available_urls
                                 if canonical_source_url(url)}

    verified_sources = list(dict.fromkeys(evidence_url_by_canonical[canonical_source_url(source)]
                                          for source in sources
                                          if canonical_source_url(source) in evidence_url_by_canonical))

    requested_year_labels = requested_years(request)

    missing_requested_years = [year
                               for year in requested_year_labels
                               if year not in labels]
    values_are_valid = (bool(labels) 
                        and bool(values)
                        and len(labels) == len(values)
                        and all(not isinstance(value, bool)
                                and isinstance(value, (int, float))
                                for value in values))

    enough_points = (len(labels) >= required_point_count(request))

    sources_are_valid = bool(verified_sources)

    valid_dataset = (data.get("can_visualize") is True
                     and values_are_valid
                     and enough_points
                     and sources_are_valid
                     and not missing_requested_years
                     and not invalid_value_labels)

    if not valid_dataset:
        missing_labels = list(dict.fromkeys([*data.get("missing_labels", []),
                                             *invalid_value_labels,
                                             *missing_requested_years]))
        reason_parts = []

        if missing_labels:
            reason_parts.append("Missing verified values for: "
                                + ", ".join(missing_labels))
        if not values_are_valid:
            reason_parts.append("The evidence did not contain matching labels "
                                "and numerical values.")
        if not enough_points:
            reason_parts.append("The evidence did not contain enough verified "
                                "data points for the requested chart.")

        if not sources_are_valid:
            reason_parts.append("The values did not have a valid source URL from "
                                "the supplied search evidence.")

        explanation = (". ".join(reason_parts)
                       or data.get("explanation")
                       or "The search evidence was insufficient for a chart.")
        return {"extraction_complete": True,
                "structured_data": {"can_visualize": False,
                                    "chart_type": requested_chart_type(request),
                                    "title": data.get("title"),
                                    "labels": labels,
                                    "values": values,
                                    "unit": data.get("unit"),
                                    "sources": [],
                                    "missing_labels": missing_labels,
                                    "explanation": explanation},
                "extraction_error": explanation}

    data.update(can_visualize=True, labels=labels, values=values, sources=verified_sources)

    if not data.get("chart_type"):
        data["chart_type"] = requested_chart_type(request)

    return {"extraction_complete": True,
            "structured_data": data,
            "extraction_error": None}


def report_writer(state: WebResearchState) -> dict:
    """Write a sourced report from the search evidence."""

    structured_data = state.get("structured_data")
    extraction_error = state.get("extraction_error")
    request = latest_request(state)

    if chart_requested(request):
        if (isinstance(structured_data, dict) and structured_data.get("can_visualize") is True):
            labels = structured_data.get("labels", [])
            values = structured_data.get("values", [])
            unit = str(structured_data.get("unit") or "").strip()
            title = (str(structured_data.get("title") or "Web research data").strip())
            value_lines = [f"- {label}: {value}{(' ' + unit) if unit else ''}"
                           for label, value in zip(labels, values)]
            source_lines = [f"- {source}"
                            for source in structured_data.get("sources", [])]
            content = "\n".join([f"**{title}**",
                                 "",
                                 *value_lines,
                                 "",
                                 "**Sources**",
                                 *source_lines]).strip()
            
        elif isinstance(structured_data, dict):
            explanation = (structured_data.get("explanation")
                           or extraction_error
                           or "The retrieved evidence was insufficient for the requested chart.")
            content = str(explanation).strip()

        else:
            content = str(extraction_error
                          or "The retrieved evidence could not be converted into verified chart data.").strip()
        return {"messages": [AIMessage(content=content)]}

    system_message = SystemMessage(content=("You are a careful web-research report writer.\n"
                                            "Answer using only the supplied search evidence and "
                                            "structured dataset.\n\n"
                                            "Rules:\n"
                                            "- Never invent or estimate numerical values.\n"
                                            "- Never claim that a chart was created.\n"
                                            "- If structured data is available, clearly list its "
                                            "labels, values, unit, and sources.\n"
                                            "- If chart data is incomplete, explain exactly which "
                                            "values are missing.\n"
                                            "- Do not say that web research failed unless searching "
                                            "actually failed.\n"
                                            "- Keep the report concise.\n"
                                            "- List the source links at the end.\n\n"
                                            f"Structured dataset:\n"
                                            f"{structured_data}\n\n"
                                            f"Extraction limitation:\n"
                                            f"{extraction_error}\n\n"
                                            f"Web-search evidence:\n"
                                            f"{state['search_results']}"))

    report = fast_model.invoke([system_message, HumanMessage(content=latest_request(state))])
    return {"messages": [report]}



builder = StateGraph(WebResearchState)

builder.add_node("sub_supervisor", sub_supervisor)
builder.add_node("researcher", researcher)
builder.add_node("data_extractor", extract_structured_data)
builder.add_node("report_writer", report_writer)

builder.add_edge(START, "sub_supervisor")
builder.add_conditional_edges("sub_supervisor", route_request, {"researcher": "researcher",
                                                                "data_extractor": "data_extractor",
                                                                "report_writer": "report_writer",
                                                                "end": END})

builder.add_edge("researcher", "sub_supervisor")
builder.add_edge("data_extractor", "sub_supervisor")
builder.add_edge("report_writer", END)

web_research_team = builder.compile()
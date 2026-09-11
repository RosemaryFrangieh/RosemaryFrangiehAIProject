import json
import re
from collections import Counter
from dataclasses import dataclass
from functools import singledispatch
from typing import Any, Literal
from pydantic import BaseModel, Field
from Models import fast_model, powerful_model
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.graph import MessagesState, StateGraph, START, END


@dataclass
class BarChart:
    title: str
    labels: list[str]
    values: list[float]

@dataclass
class PieChart:
    title: str
    labels: list[str]
    values: list[float]

@dataclass
class LineChart:
    title: str
    labels: list[str]
    values: list[float]

CHART_TYPES = {"bar": BarChart,
    	       "pie": PieChart,
    	       "line": LineChart}


@singledispatch
def create_chart(chart) -> dict:
    """Create a Chart.js configuration."""
    raise TypeError(f"Unsupported chart type: {type(chart).__name__}")


@create_chart.register
def _(chart: BarChart) -> dict:
    return {"type": "bar",
            "data": {"labels": chart.labels,
            	     "datasets": [{"label": chart.title,
                                   "data": chart.values,
                                   "backgroundColor": "#36A2EB"}]},
            "options": {"responsive": True,
                        "plugins": {"title": {"display": True,
                                              "text": chart.title}},
                        "scales": {"y": {"beginAtZero": True}}}}

@create_chart.register
def _(chart: PieChart) -> dict:
    return {"type": "pie",
            "data": {"labels": chart.labels,
                     "datasets": [{"label": chart.title,
                    	           "data": chart.values,
                    	           "backgroundColor": ["#FF6384",
                                                       "#36A2EB",
                                                       "#FFCE56",
                                                       "#4BC0C0",
                                                       "#9966FF"]}]},
            "options": {"responsive": True,
                        "plugins": {"title": {"display": True, 
                                              "text": chart.title}}}}

@create_chart.register
def _(chart: LineChart) -> dict:
    return {"type": "line",
            "data": {"labels": chart.labels,
            	     "datasets": [{"label": chart.title,
                                   "data": chart.values,
                                   "borderColor": "#36A2EB",
                                   "backgroundColor": "#36A2EB",
                                   "tension": 0.2}]},
            "options": {"responsive": True,
                        "plugins": {"title": {"display": True,
                                              "text": chart.title}},
                        "scales": {"y": {"beginAtZero": True}}}}


class PreparedChart(BaseModel):
    can_create: bool = Field(description=("True only when clear labels and numerical values exist."))
    clarification_reason: str | None = None
    chart_type: Literal["bar", "pie", "line"] | None = Field(default=None,
                                                             description="The requested Chart.js chart type.")
    title: str | None = None
    labels: list[str] = Field(default_factory=list)
    values: list[float] = Field(default_factory=list)


chart_preparation_model = powerful_model.with_structured_output(PreparedChart, method="json_mode")


def invoke_chart_preparation(preparation_instructions: str,
                             source_text: str,
                             request: str) -> PreparedChart | None:
    """Invoke the chart preparation model, self-correcting one malformed
    response.

    json_mode structured output occasionally returns a shape missing a
    required field (e.g. can_create) even when the model's underlying
    reasoning was fine. Since the model runs at temperature 0, blindly
    repeating the identical prompt tends to reproduce the identical
    mistake, so the retry includes the parser's own error as corrective
    feedback. This is generic schema repair -- it does not depend on the
    request's subject matter. Returns None only if both attempts fail,
    letting the caller fall back to a clarification response instead of
    letting the exception propagate uncaught.
    """

    messages = [SystemMessage(content=(f"{preparation_instructions}\n\n"
                                       f"Structured source data:\n{source_text}")),
                HumanMessage(content=request)]

    try:
        return chart_preparation_model.invoke(messages)
    except Exception as first_error:
        correction_messages = messages + [AIMessage(content=("My previous response could not be parsed: "
                                                             f"{first_error}")),
                                          HumanMessage(content=("Return exactly one JSON object (never a list) with "
                                                                "these top-level keys: can_create, clarification_reason, "
                                                                "chart_type, title, labels, values. Use the same "
                                                                "underlying values you already identified."))]

        try:
            return chart_preparation_model.invoke(correction_messages)
        except Exception:
            return None


class VisualizationState(MessagesState):
    source_data: dict[str, Any] | None
    prepared_chart: dict[str, Any] | None
    chart_config: dict[str, Any] | None
    visualization_status: (Literal["success", "needs_clarification"] | None)
    visualization_error: str | None


def normalize_deadline_data(source_data: dict[str, Any] | None) -> dict[str, Any] | None:
    """Convert raw task deadlines into counts per deadline."""

    if not source_data:
        return source_data

    columns = source_data.get("columns", [])
    rows = source_data.get("rows", [])
    normalized_columns = [str(column).lower() for column in columns]

    if "deadline" not in normalized_columns or not rows:
        return source_data

    count_columns = {"count", "task_count", "deadline_count", "total", "number_of_tasks"}

    if any(column in count_columns for column in normalized_columns):
        return source_data

    deadline_index = normalized_columns.index("deadline")
    deadlines = [str(row[deadline_index])
                 for row in rows
                 if (len(row) > deadline_index
                     and row[deadline_index] is not None)]

    if not deadlines:
        return source_data

    deadline_counts = Counter(deadlines)
    sorted_deadlines = sorted(deadline_counts)

    return {"can_visualize": True,
            "chart_type": "bar",
            "title": "Pending Tasks by Deadline",
            "labels": sorted_deadlines,
            "values": [deadline_counts[deadline]
                       for deadline in sorted_deadlines],
            "columns": ["deadline", "task_count"],
            "rows": [[deadline, deadline_counts[deadline]]
                     for deadline in sorted_deadlines]}

    
def prepare_chart_data(state: VisualizationState) -> dict:
    """Prepare chart fields from the request and source data."""

    if not state.get("messages"):
        return {"visualization_status": "needs_clarification",
                "visualization_error": "Please provide the chart data and chart type."}

    request = state["messages"][-1].content.strip()

    if not request:
        return {"visualization_status": "needs_clarification",
                "visualization_error": "Please provide the chart data and chart type."}

    source_data = normalize_deadline_data(state.get("source_data"))
    
    if (isinstance(source_data, dict) and source_data.get("can_visualize") is False):
        return {"prepared_chart": None,
                "visualization_status": "needs_clarification",
                "visualization_error": (source_data.get("explanation")
                                        or "The upstream data could not be verified for charting.")}

    explicit_chart_match = re.search(r"\b(bar|pie|line)\b", request, re.IGNORECASE)
    explicit_chart_type = (explicit_chart_match.group(1).lower()
                           if explicit_chart_match
                           else None)

    if source_data:
        chosen_chart_type = (explicit_chart_type or source_data.get("chart_type") or "bar")
        ready_labels = source_data.get("labels")
        ready_values = source_data.get("values")

        if (isinstance(ready_labels, list) 
            and isinstance(ready_values, list)
            and ready_labels
            and len(ready_labels) == len(ready_values)
            and all(not isinstance(value, bool)
                    and isinstance(value, (int, float))
                    for value in ready_values)):
            return {"prepared_chart": {"can_create": True,
                                       "clarification_reason": None,
                                       "chart_type": chosen_chart_type,
                                       "title": source_data.get("title") or "Data Visualization",
                                       "labels": [str(label) for label in ready_labels],
                                       "values": ready_values},
                    "visualization_status": None,
                    "visualization_error": None}

        columns = source_data.get("columns", [])
        rows = source_data.get("rows", [])

        if isinstance(columns, list) and isinstance(rows, list) and rows and len(columns) == 2:
            numeric_indexes = [index
                               for index in range(2)
                               if all(len(row) > index
                                      and not isinstance(row[index], bool)
                                      and isinstance(row[index], (int, float))
                                      for row in rows)]

            if len(numeric_indexes) == 1:
                value_index = numeric_indexes[0]
                label_index = 1 - value_index

                labels = [str(row[label_index]) for row in rows]
                values = [row[value_index] for row in rows]
                title = (f"{str(columns[value_index]).replace('_', ' ').title()} "
                         f"by {str(columns[label_index]).replace('_', ' ').title()}")

                return {"prepared_chart": {"can_create": True,
                                           "clarification_reason": None,
                                           "chart_type": chosen_chart_type,
                                           "title": title,
                                           "labels": labels,
                                           "values": values},
                        "visualization_status": None,
                        "visualization_error": None}

    preparation_instructions = """
Return one valid json object that prepares the data for one Chart.js chart.
Do not include prose or Markdown outside the json object.

Use only:
1. The user's request.
2. The structured source data provided below.

Rules:
- Never invent labels or numerical values.
- Every chart value must appear explicitly in the source data.
- Never estimate, interpolate, extrapolate, or derive missing values.
- Do not turn one overall percentage change into a yearly data series.
- If source_data contains can_visualize set to false, do not create a chart.
- If source_data already contains labels and values, preserve them exactly.
- Do not replace verified structured values with values inferred from prose.
- Use the chart_type from source_data when it exists.
- Otherwise, use the chart type explicitly requested by the user.
- Choose only bar, pie, or line.
- If there is not enough verified numerical data, set can_create to false.
- If can_create is false, explain exactly which information is missing.
- If the user does not provide a title, create a short suitable title.
- A missing title alone is not a reason to request clarification.
"""

    source_text = json.dumps(source_data, default=str)
    response = invoke_chart_preparation(preparation_instructions, source_text, request)

    if response is None or not response.can_create:
        return {"prepared_chart": None,
                "visualization_status": "needs_clarification",
                "visualization_error": ((response.clarification_reason if response else None)
                                        or ("Please provide clear chart labels "
                                            "and numerical values."))}

    labels = [str(label).strip() for label in response.labels]
    values = response.values

    if source_data:
        columns = source_data.get("columns", [])

        if response.title and response.title.strip():
            title = response.title.strip()
        elif len(columns) == 2:
            title = (f"{str(columns[1]).replace('_', ' ').title()} "
                     f"by "
                     f"{str(columns[0]).replace('_', ' ').title()}")
        else:
            title = "Data Visualization"

    if response.chart_type not in CHART_TYPES:
        error = "The chart type must be bar, pie, or line."

    elif not labels:
        error = "Chart labels are required."

    elif not values:
        error = "Chart numerical values are required."

    elif len(labels) != len(values):
        error = ("Chart labels and values must have the same length.")

    elif any(isinstance(value, bool)
             or not isinstance(value, (int, float))
             for value in values):
        error = "Every chart value must be numerical."

    else:
        prepared_chart = response.model_dump()
        prepared_chart["title"] = title
        prepared_chart["labels"] = labels
        prepared_chart["values"] = values

        return {"prepared_chart": prepared_chart,
                "visualization_status": None,
                "visualization_error": None}

    return {"prepared_chart": None,
            "visualization_status": "needs_clarification",
            "visualization_error": error}


def build_chart_config(state: VisualizationState) -> dict:
    """Create the Chart.js configuration."""

    if state.get("visualization_error"):
        return {"messages": [AIMessage(content=state["visualization_error"])],
                "visualization_status": "needs_clarification"}

    prepared_chart = state.get("prepared_chart")

    if not prepared_chart:
        error = "Valid chart data was not prepared."
        return {"messages": [AIMessage(content=error)],
            	"visualization_status": "needs_clarification",
            	"visualization_error": error}

    chart_type = prepared_chart["chart_type"]
    chart_class = CHART_TYPES[chart_type]
    chart = chart_class(title=prepared_chart["title"],
                        labels=prepared_chart["labels"],
                        values=prepared_chart["values"])

    chart_config = create_chart(chart)

    return {"messages": [AIMessage(content=(f'Created the "{prepared_chart["title"]}" '
                                            f"{chart_type} chart."))],
            "chart_config": chart_config,
            "visualization_status": "success",
            "visualization_error": None}

visualization_builder = StateGraph(VisualizationState)
visualization_builder.add_node("prepare_chart_data", prepare_chart_data)
visualization_builder.add_node("build_chart_config", build_chart_config)
visualization_builder.add_edge(START, "prepare_chart_data")
visualization_builder.add_edge("prepare_chart_data", "build_chart_config")
visualization_builder.add_edge("build_chart_config", END)
visualization_agent = visualization_builder.compile()
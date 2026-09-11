import json
import os
import re
import psycopg
from dotenv import load_dotenv, find_dotenv
from Models import fast_model
from langchain_core.messages import AIMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.graph import MessagesState, START, END, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

load_dotenv(find_dotenv())
TODO_DATABASE_URL = os.environ["TODO_DATABASE_URL"]


def initialize_database():
    """Create and populate the sample table if it does not exist."""

    with psycopg.connect(TODO_DATABASE_URL) as conn:
        with conn.cursor() as cursor:
            cursor.execute("""CREATE TABLE IF NOT EXISTS todos (
            id INTEGER GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
            task TEXT UNIQUE NOT NULL,
            deadline DATE,
            status TEXT NOT NULL)""")

            cursor.execute("""INSERT INTO todos (task, deadline, status) VALUES
            ('Finish SQL agent', '2026-08-05', 'in progress'),
            ('Prepare presentation', '2026-08-10', 'pending'),
            ('Submit project', '2026-08-15', 'pending'),
            ('Review PostgreSQL', '2026-07-25', 'completed')
            ON CONFLICT (task) DO NOTHING""")

initialize_database()


@tool
def execute_sql(query: str) -> str:
    """Execute one PostgreSQL statement.

    Supports operations such as SELECT, INSERT, UPDATE, DELETE,
    CREATE, ALTER, and DROP.

    Args:
        query: One PostgreSQL statement.
    """

    query = query.strip().removesuffix(";").strip()
    if not query:
        return json.dumps({"error": "The SQL statement is empty."})

    if ";" in query:
        return json.dumps({"error": "Only one SQL statement may be executed at a time."})

    operation_match = re.match(r"^\s*([A-Za-z]+)", query)
    operation = (operation_match.group(1).upper()
                 if operation_match
                 else "")

    if operation in {"DELETE", "DROP", "TRUNCATE"}:
        return json.dumps({"confirmation_required": True,
                           "error": ("Explicit confirmation is required before executing "
                                     f"a destructive {operation} statement.")})

    try:
        with psycopg.connect(TODO_DATABASE_URL) as conn:
            with conn.cursor() as cursor:
                cursor.execute("SET LOCAL statement_timeout = '5s'")
                cursor.execute(query)

                if cursor.description is not None:
                    columns = [column.name for column in cursor.description]
                    rows = cursor.fetchmany(51)
                    truncated = len(rows) > 50
                    rows = rows[:50]
                    result = {"operation": cursor.statusmessage,
                              "columns": columns,
                              "rows": rows,
                              "truncated": truncated,}
                else:
                    result = {"operation": cursor.statusmessage,
                              "affected_rows": cursor.rowcount,}
        return json.dumps(result, default=str)

    except psycopg.Error as error:
        return json.dumps({"error": str(error)})


model_with_tools = fast_model.bind_tools([execute_sql])

SQL_SYSTEM_MESSAGE = """
You are a PostgreSQL query agent working inside a multi-agent system.

Your job is to:
1. Determine which PostgreSQL operation the user requested.
2. Create one PostgreSQL statement.
3. Call execute_sql exactly once.
4. Do not invent query results.
5. Do not perform an operation the user did not explicitly request.

The available table is:

todos
- id: integer, generated automatically
- task: text, unique and not null
- deadline: date
- status: text, not null

This is the complete database schema available to you.

Before generating SQL, verify that the requested information can be obtained
from the listed table and columns.

If the request requires any unavailable table, entity, or column:
- Do not call execute_sql.
- Do not substitute todo records for the requested information.
- Respond exactly with:
  DATABASE_SCHEMA_UNAVAILABLE: The database contains only the todos table with
  the columns id, task, deadline, and status.
  
You may use PostgreSQL operations such as:
- SELECT
- INSERT
- UPDATE
- DELETE
- CREATE
- ALTER
- DROP

Safety rule:
- Never call execute_sql for DELETE, DROP, or TRUNCATE.
- Instead respond exactly with:
  CONFIRMATION_REQUIRED: This destructive database operation requires explicit
  confirmation before it can be executed.

Generate only one SQL statement per request.
When inserting a todo, do not provide an id because it is generated automatically.
Use RETURNING * with INSERT, UPDATE, or DELETE when the changed rows
would help explain the result.

When a request asks for counts, totals, or a distribution across a column
(e.g. "status distribution", "count by status"), return only the raw
grouped counts -- for example: SELECT status, COUNT(*) AS count FROM
todos GROUP BY status. Do not add a computed percentage column to the
query. Percentages, if needed, are calculated afterward from the raw
counts -- not computed in SQL.

When the user asks "what is on my todo list", "show my todo list",
"list my todos", or similar, return all rows from the todos table.
Do not search for the literal word "todo" in the task column.

Example:
User: What's on my todo list?
SQL: SELECT * FROM todos ORDER BY id;
"""

def generate_sql_query(state: MessagesState) -> dict[str, list]:
    """Ask the model to create and call the SQL execution tool."""
    response = model_with_tools.invoke([SystemMessage(content=SQL_SYSTEM_MESSAGE)]
                                       + state["messages"])
    return {"messages": [response]}


def explain_sql_result(state: MessagesState) -> dict[str, list]:
    """Convert the structured SQL result into a reliable answer."""
    tool_message = next((message
                         for message in reversed(state.get("messages", []))
                         if isinstance(message, ToolMessage)), None)

    if tool_message is None:
        return {"messages": [AIMessage(content="The SQL tool did not return a result.")]}

    try:
        result = json.loads(tool_message.content)

    except (TypeError, json.JSONDecodeError):
        return {"messages": [AIMessage(content=("The SQL tool returned an invalid structured result."))]}

    if result.get("error"):
        return {"messages": [AIMessage(content=f"The SQL query failed: {result['error']}")]}

    operation = str(result.get("operation", "")).upper()
    columns = result.get("columns", [])
    rows = result.get("rows")

    if rows is not None:
        if not rows:
            answer = "No matching records were found."

        else:
            formatted_rows = []

            for row in rows:
                values = [f"{column}: {value}"
                          for column, value in zip(columns, row)]
                formatted_rows.append("- " + ", ".join(values))
            answer = ("The database returned:\n"
                      + "\n".join(formatted_rows))

            if result.get("truncated"):
                answer += "\n\nOnly the first 50 records are shown."

        return {"messages": [AIMessage(content=answer)]}

    affected_rows = result.get("affected_rows")
    if affected_rows is not None:
        answer = (f"The database operation completed successfully. "
                  f"Affected rows: {affected_rows}.")

    elif operation:
        answer = (f"The database operation completed successfully: "
                  f"{operation}.")

    else:
        answer = "The database operation completed successfully."

    return {"messages": [AIMessage(content=answer)]}


sql_builder = StateGraph(MessagesState)

sql_builder.add_node("generate_sql_query", generate_sql_query)
sql_builder.add_node("tools", ToolNode([execute_sql]))
sql_builder.add_node("explain_sql_result", explain_sql_result)

sql_builder.add_edge(START, "generate_sql_query")
sql_builder.add_conditional_edges("generate_sql_query", tools_condition)
sql_builder.add_edge("tools", "explain_sql_result")
sql_builder.add_edge("explain_sql_result", END)

sql_query_agent = sql_builder.compile()
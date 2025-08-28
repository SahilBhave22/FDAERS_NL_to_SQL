# agent.py — LangGraph/validator wrapper (no DB).
import os, re, json
from typing import TypedDict, Optional, Dict, Any, Literal
from langchain_openai import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage
from langgraph.graph import StateGraph, END
import sqlglot

def make_column_inventory(catalog: Dict[str, Any]) -> str:
    lines = []
    for t in catalog.get("tables", []):
        cols = ", ".join([c["name"] for c in t.get("columns", [])])
        lines.append(f"- {t['name']}: {cols}")
    return "\\n".join(lines) if lines else "None"

def make_join_hints(catalog: Dict[str, Any]) -> str:
    rels = catalog.get("relationships", [])
    if not rels:
        return "None"
    return "\\n".join([
        f"- {r['from_table']}.{','.join(r['from_columns'])} ↔ {r['to_table']}.{','.join(r['to_columns'])} [{r.get('type','')}]"
        for r in rels
    ])

DISALLOWED = re.compile(r"\\b(insert|update|delete|drop|alter|create|copy|grant|revoke|truncate|vacuum)\\b", re.I)

def validate_sql(sql: str) -> Optional[str]:
    s = (sql or "").strip()
    if not re.match(r"^\\s*(with|select)\\b", s, flags=re.I|re.S):
        return "Only WITH/SELECT queries are allowed."
    if DISALLOWED.search(s):
        return "Disallowed SQL keyword detected."
    try:
        parsed = sqlglot.parse_one(s, read="postgres")
    except Exception as e:
        return f"SQL parse error: {e}"
    if parsed is None:
        return "Empty or unparsable SQL."
    if parsed.key not in ("SELECT", "WITH"):
        return f"Unsupported root statement: {parsed.key}"
    return None

class AgentState(TypedDict):
    question: str
    sql: Optional[str]
    error: Optional[str]
    attempts: int
    summary: Optional[str]

def build_agent(catalog: Dict[str, Any]):
    column_inventory = make_column_inventory(catalog)
    join_hints = make_join_hints(catalog)

    SYSTEM_SQL = f"""You are an expert FAERS analyst who writes clean, safe PostgreSQL.

Rules:
- Output ONE SQL query only (no commentary, no code fences).
- Read-only: WITH/SELECT only; never DDL/DML or COPY.
- Use only tables/columns that appear in the COLUMN INVENTORY below.
- Deduplicate at the report level with COUNT(DISTINCT demo.primaryid) or COUNT(DISTINCT drug_cases.primaryid).
- Prefer joins:
  - demo.primaryid ↔ drug_cases.primaryid
  - indi.(primaryid, indi_drug_seq) ↔ drug_cases.(primaryid, drug_seq)
  - reac.primaryid ↔ demo.primaryid
- Case-insensitive filters: use ILIKE for text.
- Default LIMIT 200 unless the user asks for more.
- Keep the query readable and minimal (CTEs encouraged).

SCHEMA CATALOG:
{json.dumps(catalog)}

JOIN HINTS:
{join_hints}

COLUMN INVENTORY:
{column_inventory}
"""

    SYSTEM_REVISE = f"""You are repairing a PostgreSQL query to satisfy validator feedback.

- Keep WITH/SELECT only (no DDL/DML).
- Use only tables/columns listed in this inventory:
{column_inventory}

- Apply joins consistent with these hints:
{join_hints}

- Preserve user intent; keep :snake_case parameters; end with ONE SQL query only.

Validator feedback:
{{feedback}}
"""

    llm = ChatOpenAI(model=os.getenv("FAERS_LLM_MODEL", "gpt-4o-mini"), temperature=0.1)

    def draft_sql_node(state: AgentState) -> AgentState:
        msgs = [SystemMessage(SYSTEM_SQL), HumanMessage(state["question"])]
        state["sql"] = llm.invoke(msgs).content.strip()
        return state

    def validate_sql_node(state: AgentState) -> AgentState:
        state["error"] = validate_sql(state["sql"] or "")
        return state

    def decide_next(state: AgentState) -> Literal["revise_sql", "done"]:
        return "revise_sql" if state["error"] else "done"

    def revise_sql_node(state: AgentState) -> AgentState:
        state["attempts"] += 1
        msgs = [
            SystemMessage(SYSTEM_REVISE.format(feedback=state["error"])),
            HumanMessage(state["sql"] or "")
        ]
        state["sql"] = llm.invoke(msgs).content.strip()
        state["error"] = validate_sql(state["sql"] or "")
        return state

    def done_node(state: AgentState) -> AgentState:
        return state

    graph = StateGraph(AgentState)
    graph.add_node("draft_sql", draft_sql_node)
    graph.add_node("validate_sql", validate_sql_node)
    graph.add_node("revise_sql", revise_sql_node)
    graph.add_node("done", done_node)

    graph.set_entry_point("draft_sql")
    graph.add_edge("draft_sql", "validate_sql")
    graph.add_conditional_edges("validate_sql", decide_next, {"revise_sql": "revise_sql", "done": "done"})

    def after_revise(state: AgentState):
        if state["error"] and state["attempts"] < 2:
            return "revise_sql"
        return "done"

    graph.add_conditional_edges("revise_sql", after_revise, {"revise_sql": "revise_sql", "done": "done"})
    graph.add_edge("done", END)

    app = graph.compile()
    return app

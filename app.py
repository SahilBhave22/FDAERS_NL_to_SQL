import os, json
import streamlit as st
import pandas as pd                     # NEW
from agent import build_agent           # your LangGraph builder (now executes SQL)
from urllib.parse import quote_plus


st.set_page_config(page_title="FAERS NLP → SQL (Agent Executes)", layout="wide")  # CHANGED

# --- Secrets ---
# - openai_api_key (optional; can also use env var)
# - schema_catalog_path OR schema_catalog (inline JSON string)
# - db_url (SQLAlchemy URL for your Postgres)
OPENAI_KEY = st.secrets.get("openai_api_key")


DB_PASSWORD = st.secrets.get("db_password") 
password = quote_plus(DB_PASSWORD)
DB_URL =  f"postgresql://postgres:{password}@127.0.0.1:5432/FDA_AERS"

if DB_URL:
    os.environ["DB_URL"] = DB_URL

if OPENAI_KEY:
    os.environ["OPENAI_API_KEY"] = OPENAI_KEY

@st.cache_resource
def load_catalog():
    if "schema_catalog" in st.secrets:
        try:
            return json.loads(st.secrets["schema_catalog"])
        except Exception:
            st.error("Could not parse schema_catalog in secrets.")
            raise
    path = st.secrets.get("schema_catalog_path", "schema_catalog.json")
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        st.error(f"Failed to read schema catalog at {path}.")
        raise

@st.cache_resource
def get_agent(safe_mode: bool = True, default_limit: int = 200):   # CHANGED
    if not DB_URL:
        st.error("Missing DB URL. Set `db_url` in secrets or FAERS_DB_URL/DB_URL env var.")
        st.stop()
    catalog = load_catalog()
    # build_agent now accepts db_url and returns an app that runs SQL internally
    return build_agent(catalog, db_url=DB_URL, safe_mode=safe_mode, default_limit=default_limit), catalog  # CHANGED

st.title("FAERS Natural Language → SQL → Results")  # CHANGED

#Sidebar controls (optional but useful)
with st.sidebar:                                     # NEW
    st.header("Execution Settings")
    safe_mode = st.toggle("Safe mode (SELECT/WITH only)", value=True)
    default_limit = st.number_input("Default LIMIT", 10, 10000, 200, step=10)
    # st.divider()
    # st.subheader("Schema Catalog (preview)")
    # try:
    #     st.code(json.dumps(load_catalog(), indent=2)[:900] + "...")
    # except Exception:
    #     pass

q = st.text_area(
    "Ask a FAERS question",
    placeholder="Top 10 adverse events in males for Keytruda since 2021 reported by health professionals",
    height=140,
)

go = st.button("Run", type="primary")   # CHANGED label

sql_box = st.empty()
err_box = st.empty()
attempt_box = st.empty()

if go:
    if not q.strip():
        st.error("Please enter a question.")
        st.stop()

    app, catalog = get_agent(safe_mode=safe_mode, default_limit=int(default_limit))  # CHANGED
    # agent now returns df inside state
    state = {"question": q, "sql": None, "error": None, "attempts": 0, "summary": None, "df": None}  # CHANGED
    final_state = app.invoke(state)

    sql = (final_state.get("sql") or "").strip()
    error = final_state.get("error")
    attempts = final_state.get("attempts")
    df = final_state.get("df")                              # NEW

    if sql:
        with st.expander("Generated SQL"):                  # CHANGED
            st.code(sql, language="sql")

    if error:
        err_box.error(f"Validator/Execution error: {error}")

    if isinstance(df, pd.DataFrame) and not df.empty:       # NEW
        st.success(f"Returned {len(df):,} rows · {df.shape[1]} columns")
        st.dataframe(df.head(1000), use_container_width=True, hide_index=True)
        st.download_button("Download CSV", df.to_csv(index=False).encode("utf-8"),
                           file_name="faers_results.csv")
    elif isinstance(df, pd.DataFrame):
        st.warning("Query executed, but returned 0 rows.")

    # Optional: show attempts count
    # if attempts:
    #     attempt_box.text(f"Repair attempts: {attempts}")

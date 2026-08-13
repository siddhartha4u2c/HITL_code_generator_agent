import streamlit as st
from typing import TypedDict
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import StateGraph, END
from langgraph.types import Command, interrupt
from langgraph.checkpoint.memory import MemorySaver


# -----------------------------------------------------------------------------
# LangGraph workflow (same logic as your notebook, unchanged)
# -----------------------------------------------------------------------------
class CodingAssistantState(TypedDict):
    task: str
    code: str
    tests: str


@st.cache_resource
def get_model():
    return ChatOpenAI()


@st.cache_resource
def create_coding_assistant_workflow():
    model = get_model()

    code_prompt = ChatPromptTemplate.from_template("Generate python code for : {task}")
    test_prompt = ChatPromptTemplate.from_template("Generate unit tests for this code : \n{code}")

    code_chain = code_prompt | model | StrOutputParser()
    test_chain = test_prompt | model | StrOutputParser()

    def generate_code(state):
        code = code_chain.invoke({"task": state["task"]})
        return Command(goto="human_review", update={"code": code})

    def human_review(state):
        value = interrupt({})
        if value == "yes":
            return Command(goto="create_tests")
        else:
            return Command(goto=END)

    def create_tests(state):
        tests = test_chain.invoke({"code": state["code"]})
        return Command(goto=END, update={"tests": tests})

    workflow = StateGraph(CodingAssistantState)
    workflow.add_node("generate_code", generate_code)
    workflow.add_node("human_review", human_review)
    workflow.add_node("create_tests", create_tests)
    workflow.set_entry_point("generate_code")
    return workflow.compile(checkpointer=MemorySaver())


coding_assistant = create_coding_assistant_workflow()


# ---------------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Coding Assistant", page_icon="🧑‍💻", layout="centered")
st.title("🧑‍💻 Coding Assistant")
st.caption("LangGraph workflow: generate code → human review → generate tests")

# Session state to track where we are in the workflow across Streamlit reruns
if "stage" not in st.session_state:
    st.session_state.stage = "input"        # input -> awaiting_review -> done
if "thread_id" not in st.session_state:
    st.session_state.thread_id = 1
if "code" not in st.session_state:
    st.session_state.code = ""
if "tests" not in st.session_state:
    st.session_state.tests = ""


def reset_session():
    st.session_state.thread_id += 1  # new thread so the checkpointer starts fresh
    st.session_state.stage = "input"
    st.session_state.code = ""
    st.session_state.tests = ""


thread_config = {"configurable": {"thread_id": st.session_state.thread_id}}

# --- Stage 1: get the task from the user -----------------------------------
if st.session_state.stage == "input":
    task = st.text_area(
        "What should the code do?",
        placeholder="e.g. Create a function to capitalize the first letter of a word in python",
        height=100,
    )

    if st.button("Generate code", type="primary", disabled=not task.strip()):
        with st.spinner("Generating code..."):
            result = coding_assistant.invoke({"task": task}, config=thread_config)
        st.session_state.code = result["code"]
        st.session_state.stage = "awaiting_review"
        st.rerun()

# --- Stage 2: show generated code, ask for approval ------------------------
elif st.session_state.stage == "awaiting_review":
    st.subheader("Generated code")
    st.code(st.session_state.code, language="python")

    st.write("Are you ok with this code?")
    col1, col2 = st.columns(2)

    with col1:
        if st.button("✅ Yes, generate tests", type="primary", use_container_width=True):
            with st.spinner("Generating tests..."):
                result = coding_assistant.invoke(Command(resume="yes"), config=thread_config)
            st.session_state.tests = result.get("tests", "No tests generated")
            st.session_state.stage = "done"
            st.rerun()

    with col2:
        if st.button("❌ No, stop here", use_container_width=True):
            coding_assistant.invoke(Command(resume="no"), config=thread_config)
            st.session_state.tests = ""
            st.session_state.stage = "done"
            st.rerun()

# --- Stage 3: show final result ---------------------------------------------
elif st.session_state.stage == "done":
    st.subheader("Generated code")
    st.code(st.session_state.code, language="python")

    if st.session_state.tests:
        st.subheader("Generated tests")
        st.code(st.session_state.tests, language="python")
    else:
        st.info("You rejected the code, so no tests were generated.")

    if st.button("🔁 Start over"):
        reset_session()
        st.rerun()

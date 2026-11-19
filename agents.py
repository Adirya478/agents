import os
from typing import TypedDict

import pandas as pd
import streamlit as st
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openrouter import ChatOpenRouter
from langgraph.graph import END, START, StateGraph

# ---------- Load agents from CSV ----------
CSV_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "podcast_team.csv"
)

df = pd.read_csv(CSV_PATH, skiprows=1, encoding="utf-8-sig")

df.columns = df.columns.str.strip()

required_columns = [
    "agent_name",
    "role",
    "goal",
    "instructions",
    "output"
]

missing = set(required_columns) - set(df.columns)

if missing:
    st.error(f"Missing CSV columns: {missing}")
    st.write("Available columns:", list(df.columns))
    st.stop()

AGENTS = df.to_dict("records")


# ---------- Shared state ----------
class State(TypedDict):
    user_input: str
    outputs: dict  # agent_name -> text


# ---------- Build one graph node per CSV row ----------
def make_node(agent, index, llm):
    def node(state: State) -> dict:
        outputs = dict(state["outputs"])

        if index == 0:
            context = state["user_input"]
        elif agent["agent_name"] == "Reviewer":
            context = "Original request:\n" + state["user_input"] + "\n\n"
            for name, text in outputs.items():
                context += f"--- {name} ---\n{text}\n\n"
        else:
            previous = AGENTS[index - 1]["agent_name"]
            context = outputs[previous]

        system_prompt = (
            f"You are the {agent['agent_name']}. Role: {agent['role']}.\n"
            f"Goal: {agent['goal']}\n\n"
            f"Instructions: {agent['instructions']}\n"
            f"Your output must be: {agent['output']}."
        )
        reply = llm.invoke(
            [SystemMessage(content=system_prompt), HumanMessage(content=context)]
        )
        outputs[agent["agent_name"]] = reply.content
        return {"outputs": outputs}

    return node


def build_graph(llm):
    graph = StateGraph(State)
    for i, agent in enumerate(AGENTS):
        graph.add_node(agent["agent_name"], make_node(agent, i, llm))
    graph.add_edge(START, AGENTS[0]["agent_name"])
    for i in range(len(AGENTS) - 1):
        graph.add_edge(AGENTS[i]["agent_name"], AGENTS[i + 1]["agent_name"])
    graph.add_edge(AGENTS[-1]["agent_name"], END)
    return graph.compile()


# ---------- Streamlit UI ----------
st.set_page_config(page_title="Podcast Team", page_icon="🎙️")
st.title("🎙️ Podcast Agent Team")

with st.sidebar:
    api_key = st.text_input(
        "OpenRouter API key",
        value=os.getenv("OPENROUTER_API_KEY", ""),
        type="password",
    )
    model = st.text_input("Model", value="openai/gpt-4o-mini")
    st.markdown("**Agents loaded from CSV:**")
    for a in AGENTS:
        st.write(f"• {a['agent_name']} ({a['role']})")

theme = st.text_input("Podcast theme")
audience = st.text_input("Target audience")
length = st.text_input("Episode length (e.g. 30 minutes)")

if st.button("Run agent team"):
    if not api_key:
        st.error("Please enter your OpenRouter API key.")
    elif not (theme and audience and length):
        st.error("Please fill in theme, audience and episode length.")
    else:
        os.environ["OPENROUTER_API_KEY"] = api_key
        llm = ChatOpenRouter(model="openrouter/free")
        app = build_graph(llm)
        user_input = f"Theme: {theme}\nAudience: {audience}\nEpisode length: {length}"

        with st.spinner("Agents are working..."):
            result = app.invoke({"user_input": user_input, "outputs": {}})

        for agent in AGENTS:
            with st.expander(f"{agent['agent_name']} → {agent['output']}"):
                st.markdown(result["outputs"][agent["agent_name"]])

        st.success("Final episode package")
        st.markdown(result["outputs"]["Reviewer"])
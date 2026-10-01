"""Stage 2 (LangGraph): agentic RAG as an explicit graph, with a retry loop.
Fixes the word-mismatch failure from your own tests: if search finds nothing,
an LLM rewrites the query (e.g. 'PTO' -> 'paid time off') and the graph loops back."""
from typing import TypedDict
from langchain.chat_models import init_chat_model
from langgraph.graph import StateGraph, START, END
from tools_core import search_handbook

llm = init_chat_model("google_genai:gemini-3.1-flash-lite")
MAX_ATTEMPTS = 3

class State(TypedDict):
    question: str      # what the user asked (never changes)
    query: str         # what we search for (may be rewritten)
    context: str       # passages found
    attempts: int
    answer: str

def retrieve(state: State) -> dict:
    return {"context": search_handbook(state["query"]), "attempts": state["attempts"] + 1}

def rewrite(state: State) -> dict:
    prompt = (f"A keyword search of an employee handbook for '{state['query']}' found nothing. "
              "Rewrite it as a short search query using plainer, more literal words and "
              "spelled-out acronyms. Reply with the query only.")
    return {"query": llm.invoke(prompt).text.strip()}

def generate(state: State) -> dict:
    prompt = (f"Answer using only these handbook passages and cite the page.\n\n"
              f"{state['context']}\n\nQuestion: {state['question']}")
    return {"answer": llm.invoke(prompt).text}

def route_after_retrieve(state: State) -> str:
    found = not state["context"].startswith("No matching")
    if found or state["attempts"] >= MAX_ATTEMPTS:
        return "generate"
    return "rewrite"

builder = StateGraph(State)
builder.add_node("retrieve", retrieve)
builder.add_node("rewrite", rewrite)
builder.add_node("generate", generate)
builder.add_edge(START, "retrieve")
builder.add_conditional_edges("retrieve", route_after_retrieve, ["rewrite", "generate"])
builder.add_edge("rewrite", "retrieve")          # the loop
builder.add_edge("generate", END)
graph = builder.compile()

if __name__ == "__main__":
    # print(graph.get_graph().draw_mermaid())       # paste into mermaid.live to see it
    while (q := input("\nyou> ").strip()).lower() not in {"quit", "exit"}:
        for step in graph.stream({"question": q, "query": q, "context": "", "attempts": 0, "answer": ""},
                                 stream_mode="updates"):
            node, update = next(iter(step.items()))
            print(f"  [{node}] {({k: v for k, v in update.items() if k != 'answer'})}"[:140])
            if "answer" in update:
                print("agent>", update["answer"])



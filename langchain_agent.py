from pydantic import BaseModel, Field
from langchain.agents import create_agent
from agent import rank_tool, calc_tool
from langgraph.checkpoint.memory import InMemorySaver
import uuid

MODEL="google_genai:gemini-3.5-flash"

class Answer(BaseModel):
    answer: str=Field(description="The answer in one to three sentences.")
    pages: list[str]=Field(description="Pages used form handbook eg) [p1, p3, p4]")

agent = create_agent(
    model=MODEL,
    tools=[rank_tool, calc_tool],
    system_prompt="Answer questions about the employee handbook. Always search it first. "
                  "Use calculate for any arithmetic.",
    checkpointer=InMemorySaver(),                # remembers the conversation per thread_id
    response_format=Answer, 
)

if __name__ == "__main__":
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}
    while (q := input("\nyou> ").strip()).lower() not in {"quit", "exit"}:
        result = agent.invoke({"messages": [{"role": "user", "content": q}]}, config=config)
        for m in result["messages"]:
            for call in getattr(m, "tool_calls", None) or []:
                print(f"  [tool] {call['name']}({call['args']})")
        a = result["structured_response"]
        print(f"agent> {a.answer}\n       sources: {', '.join(a.pages) or 'none'}")
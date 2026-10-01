"""Stage 3 (CrewAI): two role-played agents in a sequential pipeline.
The researcher gathers facts with your RAG tool; the writer turns them into a brief.
Covers: multi-agent hand-off, tools, task context passing."""
import os
from crewai import Agent, Task, Crew, Process, LLM
from crewai.tools import tool
from tools_core import search_handbook

llm = LLM(model="gemini/gemini-3.8-flash", api_key=os.environ["GEMINI_API_KEY"], temperature=1.0)

@tool("search_handbook")
def search_tool(query: str) -> str:
    """Search the employee handbook PDF. Returns passages tagged with page numbers."""
    return search_handbook(query)

researcher = Agent(
    role="Handbook researcher",
    goal="Find every handbook rule relevant to {topic}, with page numbers",
    backstory="You search the handbook several ways before concluding something isn't there.",
    tools=[search_tool], llm=llm, verbose=True,
)
writer = Agent(
    role="Onboarding writer",
    goal="Explain handbook rules to new hires in plain language",
    backstory="You write short, friendly briefs and never invent rules that weren't found.",
    llm=llm, verbose=True,
)

research_task = Task(
    description="Research what the handbook says about {topic}. Search at least twice with different wording.",
    expected_output="A bullet list of rules, each with its page number.",
    agent=researcher,
)
write_task = Task(
    description="Write a new-hire brief about {topic} using only the researcher's findings.",
    expected_output="A brief under 150 words with a 'Sources' line listing pages.",
    agent=writer,                      # sequential: receives research_task's output as context
)

crew = Crew(agents=[researcher, writer], tasks=[research_task, write_task],
            process=Process.sequential, verbose=True)

if __name__ == "__main__":
    topic = input("topic> ").strip() or "travel expenses"
    result = crew.kickoff(inputs={"topic": topic})
    print("\n=== BRIEF ===\n", result.raw)
from rag import rank, load_pdf
# this is a basic agent making use of registry
from google import genai
import json

client = genai.Client()


pdf_path="sample-handbook.pdf"
items=load_pdf(pdf_path)


# ---------- tool registry, Gemini's shape: {"type": "function", "name", "description", "parameters"} ----------
TOOLS = {}

def tool(description, **params):
    """Register a function as a Gemini tool. params maps argument name to JSON Schema."""
    def register(fn):
        TOOLS[fn.__name__] = {
            "fn": fn,
            "spec": {
                "type": "function",
                "name": fn.__name__,
                "description": description,
                "parameters": {
                    "type": "object",
                    "properties": params,
                    "required": list(params), # advantage 
                },
            },
        }
        return fn
    return register

@tool("Gets the value from add/sub/mul/div of two operands.",
      op1={"type": "integer", "description": "Operand 1"},
      op2={"type": "integer", "description": "Operand 2"},
      operation={"type": "string", "description": "Operation (add/sub/mul/div)"})
def calc_tool(op1, op2, operation):
    return {"add": op1 + op2, "sub": op1 - op2, "mul": op1 * op2, "div": op1 / op2}[operation]

@tool("Get chunks from uploaded pdf that is most similar to given query",
      query={"type": "string", "description": "query on pdf"},)
    #   items={"type":"array","description":"List describing the pdf read in format [(<source>,<text>)]"},
    #   k={"type":"integer", "description":"get n number of closest chunks"})
def rank_tool(query):  # items is decieded by user
    return rank(query, items, k=3)  # items and k is hardcoded default

#dispatcher: a function_call step in, a function_result dict out, never a crash ----------
def execute(step):
    try:
        if step.name not in TOOLS:
            raise ValueError(f"unknown tool '{step.name}'")
        answer = TOOLS[step.name]["fn"](**step.arguments)
        print(f"Called {step.name}({step.arguments}) -> {answer}")
        payload = {"answer": answer}
    except Exception as e:
        payload = {"error": str(e)} # advantage
    return {
        "type": "function_result",
        "name": step.name,
        "call_id": step.id,
        "result": [{"type": "text", "text": json.dumps(payload)}],
    }

history = [
    {"type": "user_input", "content": [{"type": "text", "text": "what does uploaded pdf say about expenses"}]}
]

while True:
    interaction = client.interactions.create(
        model="gemini-3.5-flash-lite",
        input=history,
        store=False,
        tools=[t["spec"] for t in TOOLS.values()],  # advantage
    )
    function_results = []
    for step in interaction.steps:
        history.append(step.model_dump())
        if step.type == "function_call":
            fn_result = execute(step)
            function_results.append(fn_result)
            history.append(fn_result)
    if not function_results:
        break

print(interaction.output_text)
"""tools_core.py: plain Python functions shared by every framework. No framework imports here."""
import ast, operator
from rag import rank, load_pdf

PDF_PATH = "data/sample-handbook.pdf"
INDEX = load_pdf(PDF_PATH)          # built once, closed over by search_handbook

def search_handbook(query: str) -> str:
    """Search the employee handbook PDF. Returns the closest passages tagged with page numbers."""
    hits = rank(query, INDEX, k=3)
    if not hits:
        return "No matching passages found."
    return "\n\n".join(f"[{source}] {text}" for _, source, text in hits)

_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
        ast.Div: operator.truediv, ast.USub: operator.neg}

def calculate(expression: str) -> str:
    """Evaluate an arithmetic expression such as '75 * 5' or '(25 - 5) / 2'."""
    def ev(node):
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
            return _OPS[type(node.op)](ev(node.left), ev(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
            return _OPS[type(node.op)](ev(node.operand))
        raise ValueError("only numbers and + - * / ( ) are allowed")
    try:
        return str(ev(ast.parse(expression, mode="eval").body))
    except Exception as e:
        return f"Error: {e}"
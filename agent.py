"""
agent.py – Agentic loop using the Anthropic API with three tools:
  1. retrieve   – contextual retrieval from the local knowledge base
  2. web_search – Anthropic-native web search tool
  3. calculator – safe arithmetic evaluation with DoS protection
"""

import ast
import math
import operator
import os

import anthropic
from dotenv import load_dotenv

from retrieval import retrieve

load_dotenv()
CLIENT = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
MODEL = "claude-haiku-4-5-20251001"

MAX_EXPONENT = 1000
MAX_RESULT_CHARS = 50

# ── Tool definitions ──────────────────────────────────────────────────

TOOLS = [
    {
        "type": "web_search_20250305",
        "name": "web_search",
    },
    {
        "name": "retrieve",
        "description": (
            "Search the local knowledge base for information relevant to the query. "
            "Returns the most relevant text chunks. Use this before web_search when "
            "the question may be answered by the knowledge base."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query.",
                }
            },
            "required": ["query"],
        },
    },
    {
        "name": "calculator",
        "description": (
            "Evaluate a mathematical expression and return the result. "
            "Supports arithmetic (+, -, *, /, **, //, %), parentheses, and common "
            "math functions: sqrt, abs, round, sin, cos, tan, log, log10, exp, pi, e. "
            "Exponents are capped at 1000."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "A Python-style math expression, e.g. '2 ** 10 + sqrt(144)'.",
                }
            },
            "required": ["expression"],
        },
    },
]

# ── Calculator implementation ─────────────────────────────────────────

_ALLOWED_NODES = (
    ast.Expression,
    ast.BinOp, ast.UnaryOp, ast.Call, ast.Constant, ast.Name,
    ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod, ast.Pow,
    ast.UAdd, ast.USub,
)

_SAFE_NAMES = {
    "sqrt": math.sqrt,
    "abs": abs,
    "round": round,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "log": math.log,
    "log10": math.log10,
    "exp": math.exp,
    "pi": math.pi,
    "e": math.e,
}

_SAFE_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}


def _eval_node(node):
    if not isinstance(node, _ALLOWED_NODES):
        raise ValueError(f"Disallowed expression: {type(node).__name__}")
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Name):
        if node.id not in _SAFE_NAMES:
            raise ValueError(f"Unknown name: {node.id}")
        return _SAFE_NAMES[node.id]
    if isinstance(node, ast.BinOp):
        op_fn = _SAFE_OPS.get(type(node.op))
        if op_fn is None:
            raise ValueError(f"Disallowed operator: {type(node.op).__name__}")
        left = _eval_node(node.left)
        right = _eval_node(node.right)
        if isinstance(node.op, ast.Pow) and right > MAX_EXPONENT:
            raise ValueError(f"Exponent too large (max {MAX_EXPONENT})")
        return op_fn(left, right)
    if isinstance(node, ast.UnaryOp):
        op_fn = _SAFE_OPS.get(type(node.op))
        if op_fn is None:
            raise ValueError(f"Disallowed unary operator: {type(node.op).__name__}")
        return op_fn(_eval_node(node.operand))
    if isinstance(node, ast.Call):
        func = _eval_node(node.func)
        args = [_eval_node(a) for a in node.args]
        return func(*args)
    raise ValueError(f"Cannot evaluate node: {type(node).__name__}")


def _format_result(result) -> str:
    if isinstance(result, float):
        if math.isnan(result):
            return "undefined (not a number)"
        if math.isinf(result):
            return "result is too large to represent"
    raw = str(result)
    if len(raw) > MAX_RESULT_CHARS:
        return f"{result:.6e}"
    if isinstance(result, int) and abs(result) >= 1000:
        return f"{result:,}"
    return raw


def safe_eval(expression: str) -> str:
    try:
        tree = ast.parse(expression.strip(), mode="eval")
        result = _eval_node(tree.body)
        return _format_result(result)
    except Exception as exc:
        return f"Error: {exc}"


# ── Tool dispatch ─────────────────────────────────────────────────────

def dispatch_tool(name: str, inputs: dict) -> str:
    if name == "calculator":
        return safe_eval(inputs["expression"])
    if name == "retrieve":
        chunks = retrieve(inputs["query"])
        if not chunks:
            return "No relevant information found in the knowledge base."
        return "\n\n---\n\n".join(c["text"] for c in chunks)
    return "Unknown tool."


# ── Agent loop ────────────────────────────────────────────────────────

def run_agent(user_message: str) -> str:
    """Run the agent loop and return the final text answer."""
    messages = [{"role": "user", "content": user_message}]

    while True:
        response = CLIENT.messages.create(
            model=MODEL,
            max_tokens=1024,
            tools=TOOLS,
            messages=messages,
        )

        messages.append({"role": "assistant", "content": response.content})

        if response.stop_reason != "tool_use":
            break

        tool_results = []
        for block in response.content:
            if block.type != "tool_use":
                continue
            if block.name in ("retrieve", "calculator"):
                result_text = dispatch_tool(block.name, block.input)
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": result_text,
                })

        if tool_results:
            messages.append({"role": "user", "content": tool_results})

    for block in response.content:
        if hasattr(block, "text"):
            return block.text.strip()

    return "I was unable to generate a response."


if __name__ == "__main__":
    query = input("Enter query: ")
    print(run_agent(query))

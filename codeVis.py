from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import ast
import sys
import copy

app = FastAPI()

# -------------------------------
# CORS (IMPORTANT for frontend)
# -------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------------
# Request Model
# -------------------------------
class CodeRequest(BaseModel):
    code: str


# -------------------------------
# STEP 1: Analyze Data Structures
# -------------------------------
class DataStructureAnalyzer(ast.NodeVisitor):
    def __init__(self):
        self.structures = set()

    def visit_List(self, node):
        self.structures.add("list")
        self.generic_visit(node)

    def visit_Dict(self, node):
        self.structures.add("dict")
        self.generic_visit(node)

    def visit_Set(self, node):
        self.structures.add("set")
        self.generic_visit(node)

    def visit_Tuple(self, node):
        self.structures.add("tuple")
        self.generic_visit(node)


# -------------------------------
# Helpers
# -------------------------------
def is_data_structure(val):
    return isinstance(val, (list, dict, set, tuple))

def is_valid_variable(var):
    return not var.startswith("__") and var != "__builtins__"


# -------------------------------
# Execution Tracer
# -------------------------------
execution_log = []
previous_state = {}

def trace(frame, event, arg):
    global previous_state

    if event == "line":
        current_vars = frame.f_locals.copy()

        for var, value in current_vars.items():

            if not is_valid_variable(var):
                continue

            if is_data_structure(value):
                prev_value = previous_state.get(var)

                # First appearance OR change
                if var not in previous_state or prev_value != value:
                    execution_log.append({
                        "line": frame.f_lineno,
                        "variable": var,
                        "type": type(value).__name__,
                        "value": copy.deepcopy(value)
                    })

        # Update state snapshot
        previous_state = {
            k: copy.deepcopy(v)
            for k, v in current_vars.items()
            if is_data_structure(v) and is_valid_variable(k)
        }

    return trace


# -------------------------------
# Core Logic Function
# -------------------------------
def analyze_and_execute(code):
    global execution_log, previous_state
    execution_log = []
    previous_state = {}

    # Analyze structures
    try:
        tree = ast.parse(code)
        analyzer = DataStructureAnalyzer()
        analyzer.visit(tree)
    except Exception as e:
        return {"error": f"Syntax Error: {str(e)}"}

    try:
        sys.settrace(trace)
        exec(code, {})   # isolated environment
        sys.settrace(None)
    except Exception as e:
        sys.settrace(None)
        return {"error": str(e)}

    return {
        "detected_structures": list(analyzer.structures),
        "execution": execution_log
    }


# -------------------------------
# API Endpoint (FINAL FIXED)
# -------------------------------
@app.post("/run-code")
def run_code(request: CodeRequest):
    return analyze_and_execute(request.code)

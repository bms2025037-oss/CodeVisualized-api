from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from RestrictedPython import compile_restricted, safe_globals, safe_builtins
import ast
import sys
import json
import signal

app = FastAPI()

# -------------------------------
# CORS — FIX 1: Removed wildcard + credentials combo
# -------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://code-visualizerbyinnoventures.netlify.app",  # deployed frontend
        "http://localhost:3000",   # local React dev
        "http://localhost:5173",   # local Vite dev
    ],
    allow_credentials=False,   # No auth/cookies needed, so False is correct
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------------
# Request Model
# -------------------------------
class CodeRequest(BaseModel):
    code: str


# -------------------------------
# STEP 1: Analyze Data Structures (AST)
# FIX 2: Now detects constructor calls like list(), dict(), set(), tuple()
# in addition to literals like [], {}, {1,2}, ()
# -------------------------------
class DataStructureAnalyzer(ast.NodeVisitor):
    def __init__(self):
        self.structures = set()

    # --- Literal detection (original) ---
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

    # --- FIX 2: Constructor call detection ---
    # Catches: list(), dict(), set(), tuple(),
    #          list([1,2]), dict(a=1), set([1,2]), tuple([1,2])
    def visit_Call(self, node):
        constructor_map = {
            "list":  "list",
            "dict":  "dict",
            "set":   "set",
            "tuple": "tuple",
        }
        # node.func is the function being called
        # ast.Name covers simple names like list(), dict()
        if isinstance(node.func, ast.Name):
            name = node.func.id
            if name in constructor_map:
                self.structures.add(constructor_map[name])

        self.generic_visit(node)


# -------------------------------
# Helpers
# -------------------------------
def is_data_structure(val):
    return isinstance(val, (list, dict, set, tuple))

def is_valid_variable(var):
    return not var.startswith("__") and var != "__builtins__"

# FIX 3: Use json.dumps for fast comparison instead of deepcopy
# Converts value to a stable string for comparison only
def make_snapshot(val):
    try:
        # sort_keys=True ensures dict comparison is order-independent
        # e.g. {"b":2,"a":1} == {"a":1,"b":2} will match correctly
        if isinstance(val, set):
            # sets are unordered — sort them before snapshotting
            return json.dumps(sorted(list(val), key=str), sort_keys=True)
        elif isinstance(val, tuple):
            return json.dumps(list(val))
        else:
            return json.dumps(val, sort_keys=True)
    except Exception:
        # fallback for non-JSON-serializable values
        return repr(val)


# -------------------------------
# Execution Tracer
# FIX 4: execution_log and previous_state are NO LONGER globals
#         They are passed in via a closure to avoid race conditions
#         between concurrent requests
# FIX 5: Line number off-by-one fixed by tracking the PREVIOUS line
#         The tracer fires BEFORE a line runs, so when tracer fires
#         at line N, it means line N-1 just finished and caused the change
# -------------------------------
def make_tracer(execution_log, previous_state):
    """
    Returns a tracer function that closes over its own
    execution_log and previous_state — isolated per request.
    """
    # We track the previously seen line number to fix the off-by-one issue
    last_line = {"value": None}

    def trace(frame, event, arg):
        if event == "line":
            current_line = frame.f_lineno
            current_vars = frame.f_locals.copy()

            for var, value in current_vars.items():
                if not is_valid_variable(var):
                    continue

                if is_data_structure(value):
                    current_snap = make_snapshot(value)
                    prev_snap    = previous_state.get(var)

                    if var not in previous_state or prev_snap != current_snap:
                        # FIX 5: Use last_line instead of current_line
                        # Because: tracer fires BEFORE line runs
                        # So changes we see now were caused by the PREVIOUS line
                        reported_line = last_line["value"] if last_line["value"] else current_line

                        execution_log.append({
                            "line":     reported_line,
                            "variable": var,
                            "type":     type(value).__name__,
                            # Store real value (not string) so frontend
                            # can directly use it for bar graph rendering
                            "value":    list(value) if isinstance(value, (set, tuple)) else value
                        })

                        # Update snapshot — fast string comparison only
                        previous_state[var] = current_snap

            # Remember current line for next tracer call
            last_line["value"] = current_line

        return trace

    return trace


# -------------------------------
# Timeout Handler
# FIX 6: Kills infinite loops — user code is limited to 5 seconds
# -------------------------------
def timeout_handler(signum, frame):
    raise TimeoutError("Code execution exceeded 5 seconds. Possible infinite loop.")


# -------------------------------
# Core Logic Function
# FIX 7: execution_log and previous_state are now LOCAL variables
#         This means each request gets its own isolated state
#         No more race conditions between concurrent users
# -------------------------------
def analyze_and_execute(code):

    # FIX 7: Local scope — not shared between requests
    execution_log  = []
    previous_state = {}

    # --- Step 1: Static AST Analysis ---
    try:
        tree     = ast.parse(code)
        analyzer = DataStructureAnalyzer()
        analyzer.visit(tree)
    except SyntaxError as e:
        return {"error": f"Syntax Error at line {e.lineno}: {e.msg}"}
    except Exception as e:
        return {"error": f"Parse Error: {str(e)}"}

    # --- Step 2: Sandbox + Execute ---
    try:
        # FIX 8: RestrictedPython sandbox
        # Compiles code in restricted mode — blocks dangerous operations:
        # - No os, sys, subprocess imports
        # - No file access
        # - No __import__ abuse
        byte_code = compile_restricted(code, filename="<user_code>", mode="exec")

        # safe_globals gives a clean, restricted global namespace
        # We add safe_builtins so basic operations (print, range, len) still work
        restricted_globals = safe_globals.copy()
        restricted_globals["__builtins__"] = safe_builtins

        # FIX 6: Set 5-second timeout before executing
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(5)

        # Install our isolated tracer (FIX 4)
        tracer = make_tracer(execution_log, previous_state)
        sys.settrace(tracer)

        exec(byte_code, restricted_globals)

        sys.settrace(None)
        signal.alarm(0)   # Cancel timeout after successful execution

    except TimeoutError as e:
        sys.settrace(None)
        signal.alarm(0)
        return {"error": str(e)}

    except Exception as e:
        sys.settrace(None)
        signal.alarm(0)
        return {"error": f"Runtime Error: {str(e)}"}

    # --- Step 3: Return results ---
    return {
        "detected_structures": list(analyzer.structures),
        "execution":           execution_log
    }


# -------------------------------
# API Endpoint
# -------------------------------
@app.post("/run-code")
def run_code(request: CodeRequest):
    return analyze_and_execute(request.code)

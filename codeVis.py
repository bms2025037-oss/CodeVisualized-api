from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from RestrictedPython import compile_restricted, safe_builtins, safe_globals
from RestrictedPython.Eval import default_guarded_getiter
from RestrictedPython.Guards import safe_globals as rp_safe_globals, guarded_iter_unpack_sequence
import ast
import sys
import json
import threading

app = FastAPI()

# -------------------------------
# CORS
# -------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://code-visualizerbyinnoventures.netlify.app",
        "http://localhost:3000",
        "http://localhost:5173",
    ],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# -------------------------------
# Request Model
# -------------------------------
class CodeRequest(BaseModel):
    code: str


# -------------------------------
# AST Analyzer
# Detects both literals [] and constructors list()
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

    def visit_Call(self, node):
        constructors = {"list", "dict", "set", "tuple"}
        if isinstance(node.func, ast.Name) and node.func.id in constructors:
            self.structures.add(node.func.id)
        self.generic_visit(node)


# -------------------------------
# Helpers
# -------------------------------
def is_data_structure(val):
    return isinstance(val, (list, dict, set, tuple))


def is_valid_variable(var):
    # Explicit blocklist — only block known internal names
    internal_names = {
        # RestrictedPython injected names
        "object",
        "args",
        "kwargs",
        "_getiter_",
        "_getitem_",
        "_write_",
        "_inplacevar_",
        "__builtins__",
        "__metaclass__",

        # Python module-level internals
        "__name__",
        "__doc__",
        "__package__",
        "__loader__",
        "__spec__",
        "__file__",
        "__cached__",
    }

    if var in internal_names:
        return False

    # Block single underscore prefix — temp/private vars
    # Do NOT block double underscore — needed by Python internals
    if var.startswith("_") and not var.startswith("__"):
        return False

    return True


def make_snapshot(val):
    """
    Converts a value to a stable string for fast comparison.
    Used only for detecting changes — not sent to frontend.
    """
    try:
        if isinstance(val, set):
            return json.dumps(sorted(list(val), key=str))
        elif isinstance(val, tuple):
            return json.dumps(list(val))
        else:
            return json.dumps(val, sort_keys=True)
    except Exception:
        return repr(val)


def serialize_value(val):
    """
    Converts value to a JSON-safe format for the frontend.
    Sets and tuples become lists so frontend bar graph can use them directly.
    """
    if isinstance(val, (set, tuple)):
        return list(val)
    return val


# -------------------------------
# Tracer Factory
# Returns an isolated tracer per request — no shared global state
# -------------------------------
def make_tracer(execution_log, previous_state, user_code):
    last_line   = {"value": None}
    total_lines = len(user_code.splitlines())

    def trace(frame, event, arg):
        if event == "line":
            current_line = frame.f_lineno

            # Filter 1: Ignore lines beyond user's code length
            if current_line > total_lines:
                return trace

            # Filter 2: Only trace frames from the user's own code
            if frame.f_code.co_filename != "<user_code>":
                return trace

            current_vars = frame.f_locals.copy()

            for var, value in current_vars.items():
                if not is_valid_variable(var):
                    continue

                if is_data_structure(value):
                    current_snap = make_snapshot(value)
                    prev_snap    = previous_state.get(var)

                    if var not in previous_state or prev_snap != current_snap:
                        reported_line = last_line["value"] if last_line["value"] else current_line

                        execution_log.append({
                            "line":     reported_line,
                            "variable": var,
                            "type":     type(value).__name__,
                            "value":    serialize_value(value)
                        })

                        previous_state[var] = current_snap

            last_line["value"] = current_line

        return trace

    return trace


# -------------------------------
# Build Restricted Globals
# We build this manually instead of using safe_globals.copy()
# so we have full control over every key — no silent overwrites
# -------------------------------
def build_restricted_globals():
    # _write_ and _getiter_ are required by RestrictedPython
    # to handle attribute writes and iteration safely
    def _write_(ob):
        return ob

    def _getiter_(ob):
        return ob

    def _getitem_(ob, index):
        return ob[index]

    def _inplacevar_(op, x, y):
        if op == "+=":  return x + y
        if op == "-=":  return x - y
        if op == "*=":  return x * y
        if op == "/=":  return x / y
        if op == "%=":  return x % y
        raise ValueError(f"Unsupported inplace op: {op}")

    return {
        # Required Python internals
        "__name__":      "__main__",   # fixes: name '__name__' is not defined
        "__doc__":       None,
        "__package__":   None,
        "__metaclass__": type,         # fixes: NameError on class creation
        "__builtins__":  safe_builtins,

        # Required RestrictedPython guard functions
        "_write_":       _write_,
        "_getiter_":     _getiter_,
        "_getitem_":     _getitem_,
        "_inplacevar_":  _inplacevar_,

        # Safe built-in types the user can use
        "list":          list,
        "dict":          dict,
        "set":           set,
        "tuple":         tuple,
        "len":           len,
        "range":         range,
        "print":         print,
        "enumerate":     enumerate,
        "zip":           zip,
        "map":           map,
        "filter":        filter,
        "sorted":        sorted,
        "reversed":      reversed,
        "sum":           sum,
        "min":           min,
        "max":           max,
        "abs":           abs,
        "round":         round,
        "isinstance":    isinstance,
        "type":          type,
        "str":           str,
        "int":           int,
        "float":         float,
        "bool":          bool,
    }


# -------------------------------
# Core Logic
# -------------------------------
def analyze_and_execute(code):

    # Local state per request — no race conditions between concurrent users
    execution_log  = []
    previous_state = {}

    # Step 1: Static AST Analysis
    try:
        tree     = ast.parse(code)
        analyzer = DataStructureAnalyzer()
        analyzer.visit(tree)
    except SyntaxError as e:
        return {"error": f"Syntax Error at line {e.lineno}: {e.msg}"}
    except Exception as e:
        return {"error": f"Parse Error: {str(e)}"}

    # Step 2: Compile with RestrictedPython sandbox
    try:
        byte_code = compile_restricted(code, filename="<user_code>", mode="exec")
    except SyntaxError as e:
        return {"error": f"Restricted Syntax Error: {str(e)}"}

    # Step 3: Execute in a thread with 5 second timeout
    restricted_globals = build_restricted_globals()
    execution_result   = {"error": None}

    def run_in_thread():
        try:
            tracer = make_tracer(execution_log, previous_state, code)
            sys.settrace(tracer)
            exec(byte_code, restricted_globals)
            sys.settrace(None)
        except Exception as e:
            sys.settrace(None)
            execution_result["error"] = f"Runtime Error: {str(e)}"

    thread = threading.Thread(target=run_in_thread)
    thread.start()
    thread.join(timeout=5)

    if thread.is_alive():
        return {"error": "Code execution exceeded 5 seconds. Possible infinite loop."}

    if execution_result["error"]:
        return {"error": execution_result["error"]}

    # Step 4: Return results
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

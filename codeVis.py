from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from RestrictedPython import compile_restricted, safe_builtins
from RestrictedPython.Guards import guarded_iter_unpack_sequence
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
    internal_names = {
        # RestrictedPython injected names
        "object",
        "args",
        "kwargs",
        "_getiter_",
        "_getitem_",
        "_write_",
        "_inplacevar_",
        "_iter_unpack_sequence_",
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
            # Prevents RestrictedPython internals leaking in
            if current_line > total_lines:
                return trace

            # Filter 2: Only trace frames from the user's own code
            # "<user_code>" is the filename set in compile_restricted()
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
                        # Off-by-one fix:
                        # Tracer fires BEFORE a line runs, so the change
                        # we see now was caused by the PREVIOUS line
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
# Built manually for full control — safe_globals.copy() silently
# ignores additions we make to it so we avoid it entirely
# -------------------------------
def build_restricted_globals():

    # Required by RestrictedPython for attribute/item writes
    # Called before every ob.attr = x or ob[i] = x
    def _write_(ob):
        return ob

    # FIX: return iter(ob) instead of ob
    # Required for all for loops — range(), list iteration etc.
    # Returning ob directly breaks range() iteration
    def _getiter_(ob):
        return iter(ob)

    # Required for index reads: ob[i]
    def _getitem_(ob, index):
        return ob[index]

    # FIX: Added missing operators **=, //=, &=, |=, ^=
    # Required for any in-place operation like arr[i] += 1
    def _inplacevar_(op, x, y):
        ops = {
            "+=":  lambda: x + y,
            "-=":  lambda: x - y,
            "*=":  lambda: x * y,
            "/=":  lambda: x / y,
            "%=":  lambda: x % y,
            "**=": lambda: x ** y,
            "//=": lambda: x // y,
            "&=":  lambda: x & y,
            "|=":  lambda: x | y,
            "^=":  lambda: x ^ y,
        }
        if op in ops:
            return ops[op]()
        raise ValueError(f"Unsupported inplace operator: {op}")

    return {
        # Required Python internals
        "__name__":      "__main__",   # fixes: name '__name__' is not defined
        "__doc__":       None,
        "__package__":   None,
        "__metaclass__": type,         # fixes: NameError on class creation
        "__builtins__":  safe_builtins,

        # Required RestrictedPython guard functions
        "_write_":                _write_,
        "_getiter_":              _getiter_,
        "_getitem_":              _getitem_,
        "_inplacevar_":           _inplacevar_,

        # FIX: Required for tuple unpacking swap: a, b = b, a
        # Without this, bubble sort swap completely fails
        "_iter_unpack_sequence_": guarded_iter_unpack_sequence,

        # Safe built-in types and functions the user can use
        "list":       list,
        "dict":       dict,
        "set":        set,
        "tuple":      tuple,
        "len":        len,
        "range":      range,
        "print":      print,
        "enumerate":  enumerate,
        "zip":        zip,
        "map":        map,
        "filter":     filter,
        "sorted":     sorted,
        "reversed":   reversed,
        "sum":        sum,
        "min":        min,
        "max":        max,
        "abs":        abs,
        "round":      round,
        "isinstance": isinstance,
        "type":       type,
        "str":        str,
        "int":        int,
        "float":      float,
        "bool":       bool,
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
    # Using threading instead of signal.SIGALRM — works on Windows + Linux
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

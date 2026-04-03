from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from RestrictedPython import compile_restricted, safe_builtins
from RestrictedPython.Guards import guarded_iter_unpack_sequence
import ast
import sys
import json
import threading
import copy

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
# Variable Usage Analyzer
# -------------------------------
class VariableUsageAnalyzer(ast.NodeVisitor):
    def __init__(self):
        self.usage = {}

    def ensure_var(self, name):
        if name not in self.usage:
            self.usage[name] = {
                "indexed": 0,
                "index_assigned": 0,
                "len_used": 0,
                "append_used": 0,
                "pop_used": 0,
                "insert_used": 0,
                "remove_used": 0,
                "extend_used": 0,
                "sort_used": 0,
            }

    def visit_Subscript(self, node):
        if isinstance(node.value, ast.Name):
            name = node.value.id
            self.ensure_var(name)
            self.usage[name]["indexed"] += 1
        self.generic_visit(node)

    def visit_Assign(self, node):
        for target in node.targets:
            if isinstance(target, ast.Subscript) and isinstance(target.value, ast.Name):
                name = target.value.id
                self.ensure_var(name)
                self.usage[name]["index_assigned"] += 1
        self.generic_visit(node)

    def visit_AugAssign(self, node):
        if isinstance(node.target, ast.Subscript) and isinstance(node.target.value, ast.Name):
            name = node.target.value.id
            self.ensure_var(name)
            self.usage[name]["index_assigned"] += 1
        self.generic_visit(node)

    def visit_Call(self, node):
        # len(arr)
        if isinstance(node.func, ast.Name) and node.func.id == "len":
            if node.args and isinstance(node.args[0], ast.Name):
                name = node.args[0].id
                self.ensure_var(name)
                self.usage[name]["len_used"] += 1

        # arr.append(), arr.pop(), ...
        if isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name):
            name = node.func.value.id
            method = node.func.attr
            self.ensure_var(name)

            if method == "append":
                self.usage[name]["append_used"] += 1
            elif method == "pop":
                self.usage[name]["pop_used"] += 1
            elif method == "insert":
                self.usage[name]["insert_used"] += 1
            elif method == "remove":
                self.usage[name]["remove_used"] += 1
            elif method == "extend":
                self.usage[name]["extend_used"] += 1
            elif method == "sort":
                self.usage[name]["sort_used"] += 1

        self.generic_visit(node)


# -------------------------------
# Helpers
# -------------------------------
def is_data_structure(val):
    return isinstance(val, (list, dict, set, tuple))


def is_valid_variable(var):
    internal_names = {
        "object", "args", "kwargs",
        "_getiter_", "_getitem_", "_write_", "_inplacevar_",
        "_iter_unpack_sequence_",
        "__builtins__", "__metaclass__",
        "__name__", "__doc__", "__package__",
        "__loader__", "__spec__", "__file__", "__cached__",
    }

    if var in internal_names:
        return False

    if var.startswith("_") and not var.startswith("__"):
        return False

    return True


def make_snapshot(val):
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
    if isinstance(val, (set, tuple)):
        return list(val)
    return val


def normalize_python_type(val):
    if isinstance(val, list):
        return "list"
    if isinstance(val, dict):
        return "dict"
    if isinstance(val, set):
        return "set"
    if isinstance(val, tuple):
        return "tuple"
    return type(val).__name__


def detect_visual_type(var_name, value, usage_map):
    python_type = normalize_python_type(value)

    if python_type != "list":
        return python_type

    usage = usage_map.get(var_name, {})

    array_score = (
        usage.get("indexed", 0)
        + usage.get("index_assigned", 0)
        + usage.get("len_used", 0)
    )

    list_score = (
        usage.get("append_used", 0)
        + usage.get("pop_used", 0)
        + usage.get("insert_used", 0)
        + usage.get("remove_used", 0)
        + usage.get("extend_used", 0)
    )

    if array_score > list_score:
        return "array"
    if list_score > array_score:
        return "list"

    # Default for DSA-style code like sorting/searching
    return "array"


# -------------------------------
# Tracer
# -------------------------------
def make_tracer(execution_log, previous_state, user_code, usage_map):
    total_lines = len(user_code.splitlines())

    def trace(frame, event, arg):
        if event == "line":
            current_line = frame.f_lineno

            if current_line > total_lines:
                return trace

            if frame.f_code.co_filename != "<user_code>":
                return trace

            current_vars = frame.f_locals.copy()

            for var, value in current_vars.items():
                if not is_valid_variable(var):
                    continue

                if is_data_structure(value):
                    current_snap = make_snapshot(value)
                    prev_snap = previous_state.get(var)

                    if var not in previous_state:
                        previous_state[var] = current_snap
                        execution_log.append({
                            "line": current_line,
                            "variable": var,
                            "python_type": normalize_python_type(value),
                            "visual_type": detect_visual_type(var, value, usage_map),
                            "value": copy.deepcopy(serialize_value(value))
                        })
                        continue

                    if prev_snap != current_snap:
                        execution_log.append({
                            "line": current_line,
                            "variable": var,
                            "python_type": normalize_python_type(value),
                            "visual_type": detect_visual_type(var, value, usage_map),
                            "value": copy.deepcopy(serialize_value(value))
                        })

                        previous_state[var] = current_snap

        return trace

    return trace


# -------------------------------
# Restricted Globals
# -------------------------------
def build_restricted_globals(output_buffer):

    def _write_(ob):
        return ob

    def _getiter_(ob):
        return iter(ob)

    def _getitem_(ob, index):
        return ob[index]

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
        raise ValueError(f"Unsupported operator: {op}")

    def safe_print(*args, **kwargs):
        output_buffer.append(" ".join(map(str, args)))

    return {
        "__name__": "__main__",
        "__doc__": None,
        "__package__": None,
        "__metaclass__": type,
        "__builtins__": safe_builtins,

        "_write_": _write_,
        "_getiter_": _getiter_,
        "_getitem_": _getitem_,
        "_inplacevar_": _inplacevar_,
        "_iter_unpack_sequence_": guarded_iter_unpack_sequence,

        "list": list, "dict": dict, "set": set, "tuple": tuple,
        "len": len, "range": range, "enumerate": enumerate,
        "zip": zip, "map": map, "filter": filter,
        "sorted": sorted, "reversed": reversed,
        "sum": sum, "min": min, "max": max,
        "abs": abs, "round": round,
        "isinstance": isinstance, "type": type,
        "str": str, "int": int, "float": float, "bool": bool,

        "print": safe_print,
    }


# -------------------------------
# Core Logic
# -------------------------------
def analyze_and_execute(code):

    execution_log = []
    previous_state = {}
    output_buffer = []

    try:
        tree = ast.parse(code)

        analyzer = DataStructureAnalyzer()
        analyzer.visit(tree)

        usage_analyzer = VariableUsageAnalyzer()
        usage_analyzer.visit(tree)
        usage_map = usage_analyzer.usage

    except SyntaxError as e:
        return {"error": f"Syntax Error at line {e.lineno}: {e.msg}"}
    except Exception as e:
        return {"error": f"Parse Error: {str(e)}"}

    try:
        byte_code = compile_restricted(code, filename="<user_code>", mode="exec")
    except SyntaxError as e:
        return {"error": f"Restricted Syntax Error: {str(e)}"}

    restricted_globals = build_restricted_globals(output_buffer)
    execution_result = {"error": None}

    def run():
        try:
            tracer = make_tracer(execution_log, previous_state, code, usage_map)
            sys.settrace(tracer)
            exec(byte_code, restricted_globals)
            sys.settrace(None)
        except Exception as e:
            sys.settrace(None)
            execution_result["error"] = f"Runtime Error: {str(e)}"

    thread = threading.Thread(target=run)
    thread.start()
    thread.join(timeout=5)

    if thread.is_alive():
        return {"error": "Execution timeout (possible infinite loop)"}

    if execution_result["error"]:
        return {"error": execution_result["error"]}

    return {
        "detected_structures": list(analyzer.structures),
        "execution": execution_log,
        "output": output_buffer
    }


# -------------------------------
# API
# -------------------------------
@app.post("/run-code")
def run_code(request: CodeRequest):
    return analyze_and_execute(request.code)

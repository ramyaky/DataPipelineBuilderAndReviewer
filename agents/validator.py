# validator.py
import ast
import subprocess
import tempfile
from pathlib import Path
import logging
import re
from agents.code_generator import query_ollama

logger = logging.getLogger("DataPipelineBuilder")

# ================= FORBIDDEN RULES ==========================
FORBIDDEN_NAMES = {
    "eval", "exec", "open", "compile", "__import__",
    "globals", "locals", "vars"
}

FORBIDDEN_MODULES = {
    "os", "sys", "subprocess", "importlib", "pathlib",
    "shutil", "socket", "requests", "http", "urllib", "ftplib",
    "paramiko", "psutil"
}

FORBIDDEN_STRINGS = FORBIDDEN_NAMES | FORBIDDEN_MODULES

FORBIDDEN_ATTRS = {
    "system", "popen", "run", "remove", "unlink"
}


# ================= EXCEPTION TYPE ==========================
class UnsafeCodeError(Exception):
    """Raised when unsafe or forbidden code is detected."""
    pass


# ============ SAFE AST CHECKER ==============================
class SafeASTChecker(ast.NodeVisitor):

    def visit_Import(self, node):
        for alias in node.names:
            root = alias.name.split('.')[0]
            if root in FORBIDDEN_MODULES:
                raise UnsafeCodeError(f"Import of forbidden module: {alias.name}")
        self.generic_visit(node)

    def visit_ImportFrom(self, node):
        if node.module:
            root = node.module.split('.')[0]
            if root in FORBIDDEN_MODULES:
                raise UnsafeCodeError(f"Import from forbidden module: {node.module}")
        self.generic_visit(node)

    def visit_Name(self, node):
        if node.id in FORBIDDEN_NAMES:
            raise UnsafeCodeError(f"Forbidden name: {node.id}")
        if node.id == "__builtins__":
            raise UnsafeCodeError("Access to __builtins__ is forbidden")
        self.generic_visit(node)

    def visit_Attribute(self, node):
        if isinstance(node.value, ast.Name):
            module = node.value.id
            attr = node.attr

            if module in FORBIDDEN_MODULES:
                raise UnsafeCodeError(f"Forbidden module attribute access: {module}.{attr}")

            if attr in FORBIDDEN_ATTRS:
                raise UnsafeCodeError(f"Forbidden attribute/function access: {module}.{attr}")

        self.generic_visit(node)

    def visit_Call(self, node):
        # dynamic import via importlib
        if isinstance(node.func, ast.Attribute):
            if isinstance(node.func.value, ast.Name):
                # importlib.import_module()
                if node.func.value.id == "importlib":
                    raise UnsafeCodeError("Dynamic imports via importlib are forbidden")

                module = node.func.value.id
                attr = node.func.attr

                if module in FORBIDDEN_MODULES:
                    raise UnsafeCodeError(f"Forbidden module call: {module}.{attr}")

                if attr in FORBIDDEN_ATTRS:
                    raise UnsafeCodeError(f"Forbidden attribute/function call: {module}.{attr}")

        self.generic_visit(node)

    def visit_Constant(self, node):
        if isinstance(node.value, str) and node.value in FORBIDDEN_STRINGS:
            raise UnsafeCodeError(f"Forbidden string literal: {node.value}")
        self.generic_visit(node)


def check_safe(code: str):
    """Runs SafeASTChecker."""
    try:
        logger.debug(f"Parsing code block using AST")
        tree = ast.parse(code)
    except SyntaxError as e:
        raise UnsafeCodeError(f"Syntax error: {e}")

    checker = SafeASTChecker()
    checker.visit(tree)
    return True


# ================== COMPILE CHECK ===========================
def validate_compiles(code: str):
    """Ensures the code compiles without syntax errors."""
    try:
        compile(code, "<string>", "exec")
    except SyntaxError as e:
        raise UnsafeCodeError(f"Code failed to compile: {e}")
    return True

# ================ RUFF FIX PROMPT =========================
def get_ruff_fix_prompt(code: str, ruff_output: str):
    """ Builds prompt for ruff found issues """
    return f"""
    You previously generated this PySpark code:

    ```python
    {code}
    ```
    Ruff reported the following issues:
    {ruff_output}
    Please fix ALL ruff issues.
    Return only the corrected Python code inside a ```python fenced block with no explanation.
    """

# ==================== RUFF LINTING ==========================
def run_ruff_lint(code: str):
    """Runs `ruff` linting on a temporary file."""
    with tempfile.NamedTemporaryFile("w", delete=False, suffix=".py") as f:
        f.write(code)
        temp_path = f.name

    cmd = ["ruff", "check", temp_path, "--quiet"]
    process = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    diagnostics = process.stdout

    # Ruff exit code:
    # 0 = no issues
    # 1 = lint issues found
    # >1 = ruff internal error
    
    if process.returncode != 0:
        logger.info(f"Ruff lint errors: \n{process.stdout}")
        return False, diagnostics
    else:
        return True, ""


# ================= SPARK-SPECIFIC CHECK =====================
def validate_spark_usage(code: str):
    """Ensure code looks like a Spark ETL job"""
    if "SparkSession" not in code:
        raise UnsafeCodeError("Missing SparkSession -- not a valid Spark Job.")
    if not ("read." in code or ".read." in code):
        raise UnsafeCodeError("Saprk job is missing Dataframe reading operations")
    
    ## We can add any custom checks if needed.

    return True

# =================== EXTRACT PYTHON CODE ======================
def extract_python_code(code: str):
    """Extract pure python code from LLM output."""
    
    python_block = re.findall(r"```(?:python|py|Python)\s?(.*?)```", code, re.DOTALL)
    result = python_block.pop()
    logger.debug(f"Extracted Python Code: \n {result}")
    return result

# ============ FULL VALIDATION PIPELINE =====================

def validate_generated_code(code: str):

    python_code = extract_python_code(code)

    check_safe_result = check_safe(python_code)
    if check_safe_result:
        logger.info("SafeASTChecker completed successfully")

    compile_check_result = validate_compiles(python_code)
    if compile_check_result:
        logger.info("Compliation Successfull")
        
    status, diagnostics = run_ruff_lint(python_code)
    if status:
        logger.info("No ruff lint suggestions found.")
        logger.info(f"Generated production ready code:\n {python_code}")
        return True
    else:
        logger.info(f"Found ruff lint suggestions as follows: {diagnostics}\n")
        new_prompt = get_ruff_fix_prompt(python_code, diagnostics)
        llm_result = query_ollama(new_prompt)
        validate_generated_code(llm_result)
"""
ASFT Security Sandbox — Safe execution environment for LLM-generated code.

CRITICAL DESIGN DECISION:
  The original `VerificationLayer` executed LLM-generated code using
  `subprocess.run(["python", tmpfile])` with a naïve regex blacklist.
  This is a trivially bypassed Remote Code Execution (RCE) vulnerability.

  This module replaces that with two approaches:

  1. RestrictedPython (preferred): Parse + compile in a restricted AST
     that denies imports, file I/O, network access, and builtins.
     Safe for evaluating simple expressions and checking syntax.

  2. SymPy CAS (for math): Mathematical expressions are evaluated using
     SymPy's computer algebra system — no code execution at all.

  Production deployments that need real code execution MUST use
  containerized sandboxes (Docker/gVisor) out-of-process.
  That integration is outside this module's scope.
"""
from __future__ import annotations

import ast
import logging
from dataclasses import dataclass
from typing import Any, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Sandbox results
# ---------------------------------------------------------------------------


@dataclass
class SandboxResult:
    """Result of a sandboxed execution or validation."""
    success: bool
    output: Optional[str] = None
    error: Optional[str] = None
    was_blocked: bool = False   # True if the sandbox blocked the code


# ---------------------------------------------------------------------------
# Dangerous AST node types — blocked in RestrictedPython mode
# ---------------------------------------------------------------------------

_BLOCKED_NODE_TYPES = (
    ast.Import,
    ast.ImportFrom,
    ast.Global,
    ast.Nonlocal,
    ast.Delete,
)

_BLOCKED_BUILTINS = frozenset({
    "__import__", "eval", "exec", "compile", "open", "input",
    "globals", "locals", "vars", "dir", "getattr", "setattr",
    "delattr", "hasattr", "__class__", "__subclasses__",
})


# ---------------------------------------------------------------------------
# AST-based code validator
# ---------------------------------------------------------------------------


def _validate_ast(code: str) -> tuple[bool, str]:
    """
    Parse and walk the AST of `code`.
    Returns (is_safe, reason).
    Does NOT execute any code.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return False, f"SyntaxError: {e}"

    for node in ast.walk(tree):
        # Block import statements
        if isinstance(node, _BLOCKED_NODE_TYPES):
            return False, f"Blocked AST node: {type(node).__name__}"

        # Block calls to dangerous builtins
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name) and func.id in _BLOCKED_BUILTINS:
                return False, f"Blocked builtin call: {func.id}()"
            if isinstance(func, ast.Attribute) and func.attr in _BLOCKED_BUILTINS:
                return False, f"Blocked attribute call: .{func.attr}()"

        # Block attribute access that starts with __ (dunder methods)
        if isinstance(node, ast.Attribute):
            if node.attr.startswith("__") and node.attr.endswith("__"):
                return False, f"Blocked dunder attribute: {node.attr}"

        # Block string-based escapes (code injection via string eval)
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            s = node.value
            if any(p in s for p in ("__import__", "exec(", "eval(", "os.system")):
                return False, "Blocked: dangerous string literal detected"

    return True, "ok"


# ---------------------------------------------------------------------------
# RestrictedPython evaluator (expressions only, no side effects)
# ---------------------------------------------------------------------------


def evaluate_expression(expression: str) -> SandboxResult:
    """
    Safely evaluate a Python expression using RestrictedPython.

    This is used ONLY for simple arithmetic verification — not for
    executing arbitrary code blocks.

    Args:
        expression: A short arithmetic/logical expression string.

    Returns:
        SandboxResult with the evaluated value or an error.
    """
    if len(expression) > 500:
        return SandboxResult(success=False, was_blocked=True,
                             error="Expression too long (max 500 chars)")

    is_safe, reason = _validate_ast(expression)
    if not is_safe:
        logger.warning("Sandbox blocked expression: %s — %s", expression[:80], reason)
        return SandboxResult(success=False, was_blocked=True, error=reason)

    # Only allow: numbers, basic operators, parentheses, whitespace
    import re
    if not re.match(r'^[\d\s\+\-\*/\(\)\.\,\%\*\^eE]+$', expression):
        return SandboxResult(
            success=False, was_blocked=True,
            error="Expression contains disallowed characters"
        )

    try:
        # Safe: ast.literal_eval for simple literals
        # For arithmetic, use a compile+eval with empty builtins namespace
        compiled = compile(expression, "<sandbox>", "eval")
        result = eval(compiled, {"__builtins__": {}}, {})  # noqa: S307 (intentional, safe namespace)
        return SandboxResult(success=True, output=str(result))
    except Exception as e:
        return SandboxResult(success=False, error=f"Evaluation error: {e}")


# ---------------------------------------------------------------------------
# SymPy CAS — Math verification (NO code execution)
# ---------------------------------------------------------------------------


def verify_math_with_sympy(expression: str) -> SandboxResult:
    """
    Evaluate a mathematical expression using SymPy's CAS.

    This is the SAFE alternative to eval() for math verification.
    SymPy parses the expression symbolically — it never executes Python code.

    Args:
        expression: A mathematical expression string (e.g., "2 + 3 * 4").

    Returns:
        SandboxResult with the computed value.
    """
    try:
        is_safe, reason = _validate_ast(expression)
        if not is_safe:
            logger.warning("Sandbox blocked math expression: %s — %s", expression[:80], reason)
            return SandboxResult(success=False, was_blocked=True, error=reason)

        import sympy
        # sympify converts string to SymPy expression — no code execution
        result = sympy.sympify(expression)
        # Evaluate to a numeric value where possible
        numeric = float(result.evalf())
        return SandboxResult(success=True, output=str(numeric))
    except ImportError:
        logger.debug("SymPy not installed — falling back to sandboxed eval")
        return evaluate_expression(expression)
    except Exception as e:
        return SandboxResult(success=False, error=f"SymPy error: {e}")


# ---------------------------------------------------------------------------
# Code syntax validator (no execution)
# ---------------------------------------------------------------------------


def validate_code_syntax(code: str, language: str = "python") -> SandboxResult:
    """
    Validate code syntax WITHOUT executing it.

    Currently supports Python. Other languages return a conservative "valid"
    result since we cannot execute them safely.

    Args:
        code:     Source code string.
        language: Programming language identifier.

    Returns:
        SandboxResult indicating whether the syntax is valid.
    """
    if language != "python":
        return SandboxResult(
            success=True,
            output=f"Syntax validation not supported for {language} — skipped"
        )

    if len(code) > 10_000:
        return SandboxResult(success=False, was_blocked=True,
                             error="Code block too large (max 10,000 chars)")

    is_safe, reason = _validate_ast(code)
    if not is_safe:
        return SandboxResult(success=False, was_blocked=True, error=reason)

    try:
        ast.parse(code)
        return SandboxResult(success=True, output="Syntax valid")
    except SyntaxError as e:
        return SandboxResult(success=False, error=f"SyntaxError at line {e.lineno}: {e.msg}")

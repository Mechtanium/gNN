"""Every name the package references at runtime is defined.

The package is a trimmed copy of a research code base; a trim that deletes a
helper whose caller survives is invisible until a deck reaches that branch
(``_parse_wconinje_row`` was gone for every deck with a WCONINJE block while
SPE-2, which has none, ran fine). This is the F821 check, in the test suite so
it runs wherever the tests do.
"""
import ast
import builtins
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
FILES = sorted((ROOT / "modules").rglob("*.py")) + [ROOT / "workflow.py", ROOT / "page.py"]
BUILTINS = set(dir(builtins)) | {"__file__", "__name__", "__doc__", "__spec__", "__path__"}


def _bound_names(node: ast.AST) -> set[str]:
    """Names bound directly in this scope (not in nested function/class bodies)."""
    out: set[str] = set()

    def visit(n: ast.AST, top: bool) -> None:
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            out.add(n.name)
            if not top:
                return
            if isinstance(n, ast.ClassDef):
                for b in n.body:
                    visit(b, False)
                return
            a = n.args
            for arg in a.posonlyargs + a.args + a.kwonlyargs:
                out.add(arg.arg)
            if a.vararg:
                out.add(a.vararg.arg)
            if a.kwarg:
                out.add(a.kwarg.arg)
            for b in n.body:
                visit(b, False)
            return
        if isinstance(n, ast.Lambda):
            if not top:
                return
            a = n.args
            for arg in a.posonlyargs + a.args + a.kwonlyargs:
                out.add(arg.arg)
            if a.vararg:
                out.add(a.vararg.arg)
            if a.kwarg:
                out.add(a.kwarg.arg)
            visit(n.body, False)
            return
        if isinstance(n, (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)):
            # comprehension targets bind inside the comprehension only; treat as bound
            # here (a superset is safe: we only want to avoid false positives).
            for g in n.generators:
                for t in ast.walk(g.target):
                    if isinstance(t, ast.Name):
                        out.add(t.id)
        if isinstance(n, ast.Name) and isinstance(n.ctx, (ast.Store, ast.Del)):
            out.add(n.id)
        if isinstance(n, (ast.Import, ast.ImportFrom)):
            for alias in n.names:
                out.add((alias.asname or alias.name).split(".")[0])
        if isinstance(n, ast.ExceptHandler) and n.name:
            out.add(n.name)
        if isinstance(n, (ast.Global, ast.Nonlocal)):
            out.update(n.names)
        if isinstance(n, ast.arg):
            out.add(n.arg)
        for c in ast.iter_child_nodes(n):
            visit(c, False)

    visit(node, True)
    return out


def _undefined(tree: ast.Module) -> list[tuple[int, str]]:
    module_names = _bound_names(tree) | BUILTINS
    bad: list[tuple[int, str]] = []

    def check_scope(node: ast.AST, visible: set[str]) -> None:
        scope = visible | _bound_names(node)
        for child in ast.walk(node):
            if isinstance(child, ast.Name) and isinstance(child.ctx, ast.Load) and child.id not in scope:
                bad.append((child.lineno, child.id))
        # nested scopes see everything bound here
        for child in ast.walk(node):
            if child is not node and isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
                pass  # walked above with the enclosing scope's names already included

    check_scope(tree, module_names)
    # ast.walk from the module already visits nested functions; their locals are
    # collected by _bound_names(tree) only at module level, so gather per function.
    for fn in ast.walk(tree):
        if isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            local = _bound_names(fn)
            for child in ast.walk(fn):
                if isinstance(child, ast.Name) and isinstance(child.ctx, ast.Load) and child.id in local:
                    bad[:] = [b for b in bad if not (b[0] == child.lineno and b[1] == child.id)]
    return sorted(set(bad))


@pytest.mark.parametrize("path", FILES, ids=lambda p: str(p.relative_to(ROOT)))
def test_no_undefined_names(path: Path):
    tree = ast.parse(path.read_text(), filename=str(path))
    bad = _undefined(tree)
    assert not bad, f"undefined names in {path.relative_to(ROOT)}: {bad}"

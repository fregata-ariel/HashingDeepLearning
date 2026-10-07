"""Static integration check for CLI scheduler configuration propagation."""
from __future__ import annotations
import ast
from pathlib import Path

# TRACE_TEST_ID: MONGOOSE-CONFIG-PROPAGATION

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "ports/mongoose/mongoose_reformer/train_reformer.py"

def test_scheduler_cli_values_reach_model_constructor() -> None:
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
    calls = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "ReformerLM_tune"
    ]
    assert len(calls) == 1
    keywords = {kw.arg: ast.unparse(kw.value) for kw in calls[0].keywords if kw.arg}
    assert keywords.get("scheduler_hashes") == "args.scheduler_hashes"
    assert keywords.get("thresh") == "args.thresh"

if __name__ == "__main__":
    test_scheduler_cli_values_reach_model_constructor()
    print("TRACE_TEST_PASS MONGOOSE-CONFIG-PROPAGATION")

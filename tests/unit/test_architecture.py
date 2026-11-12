import ast
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import ev_assignment


def test_core_has_no_orchestration_dependency_or_machine_paths():
    root = Path(ev_assignment.__file__).parent
    for source in root.rglob("*.py"):
        text = source.read_text()
        assert "/home/Shixin" not in text and "/Users/" not in text, source
        assert "sys.path" not in text and "spec_from_file_location" not in text, source
        for node in ast.walk(ast.parse(text)):
            if isinstance(node, ast.ImportFrom):
                assert not (node.module or "").startswith(
                    ("evcs_experiments", "ev_assignment_experiments")
                ), source
            if isinstance(node, ast.Import):
                assert all(
                    not a.name.startswith(("evcs_experiments", "ev_assignment_experiments"))
                    for a in node.names
                ), source


def test_import_from_outside_checkout_does_not_load_gurobi(tmp_path):
    code = """import sys
import ev_assignment.models.first_stage
import ev_assignment.models.recourse
import ev_assignment.methods.probability
import ev_assignment.l08.energy
import ev_assignment.l08.physics
import ev_assignment.l08.matrix
import ev_assignment.l08.reduction
import ev_assignment_experiments.cli
assert 'gurobipy' not in sys.modules
"""
    subprocess.run([sys.executable, "-c", code], cwd=tmp_path, check=True)
    assert list(tmp_path.iterdir()) == []


def test_migrated_numerical_ast_matches_baseline(pytestconfig):
    # pytest owns the checkout location. This is source migration verification,
    # not a scientific dataset root inferred from package location.
    root = Path(pytestconfig.rootpath)
    manifest = json.loads((root / "docs/architecture/scientific-ast-manifest.json").read_text())

    class NoImports(ast.NodeTransformer):
        def visit_Import(self, node):
            return None

        def visit_ImportFrom(self, node):
            return None

    for path, expected in manifest["files"].items():
        tree = NoImports().visit(ast.parse((root / path).read_text()))
        digest = hashlib.sha256(ast.dump(tree, include_attributes=False).encode()).hexdigest()
        assert digest == expected, path


def test_l08_source_and_extraction_manifest(pytestconfig):
    root = Path(pytestconfig.rootpath)
    manifest = json.loads((root / "docs/architecture/l08-extraction-manifest.json").read_text())
    for row in manifest["files"]:
        for role in ("source", "target"):
            assert (
                hashlib.sha256(
                    (
                        (root / "reference/baseline" / row[role])
                        if role == "source"
                        else (root / row[role])
                    ).read_bytes()
                ).hexdigest()
                == row[role + "_sha256"]
            )


def test_baseline_source_bytes_preserved(pytestconfig):
    root = Path(pytestconfig.rootpath)
    manifest = json.loads((root / "docs/architecture/baseline-source-manifest.json").read_text())
    for row in manifest["files"]:
        assert (
            hashlib.sha256((root / "reference/baseline" / row["path"]).read_bytes()).hexdigest()
            == row["sha256"]
        )


def test_native_numerical_functions_keep_original_ast(pytestconfig):
    import copy

    root = Path(pytestconfig.rootpath)
    manifest = json.loads(
        (root / "docs/architecture/l08-native-migration-manifest.json").read_text()
    )

    class Normalize(ast.NodeTransformer):
        def visit_Import(self, node):
            return None

        def visit_ImportFrom(self, node):
            return None

        def visit_Assign(self, node):
            if any(isinstance(t, ast.Name) and t.id == "CFG" for t in node.targets):
                return None
            return self.generic_visit(node)

    for row in manifest["files"]:
        for role in ("source", "target"):
            assert (
                hashlib.sha256(
                    (
                        (root / "reference/baseline" / row[role])
                        if role == "source"
                        else (root / row[role])
                    ).read_bytes()
                ).hexdigest()
                == row[role + "_sha256"]
            )
        old = {
            n.name: n
            for n in ast.parse((root / "reference/baseline" / row["source"]).read_text()).body
            if isinstance(n, ast.FunctionDef)
        }
        new = {
            n.name: n
            for n in ast.parse((root / row["target"]).read_text()).body
            if isinstance(n, ast.FunctionDef)
        }
        for name in row.get("functions", []):
            assert ast.dump(
                Normalize().visit(copy.deepcopy(old[name])), include_attributes=False
            ) == ast.dump(Normalize().visit(copy.deepcopy(new[name])), include_attributes=False), (
                row["target"],
                name,
            )


def test_all_core_modules_import_without_solver_or_io(tmp_path):
    source = """import importlib,pkgutil,sys
import ev_assignment
for m in pkgutil.walk_packages(ev_assignment.__path__,ev_assignment.__name__+'.'):
 importlib.import_module(m.name)
assert 'gurobipy' not in sys.modules
"""
    subprocess.run([sys.executable, "-c", source], cwd=tmp_path, check=True)
    assert list(tmp_path.iterdir()) == []

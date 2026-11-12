"""Build an installable immutable release from the exact recorded source archive."""

import io
import json
from pathlib import Path
import zipfile
from ev_assignment_experiments.tracking.store import encode, sha256
from ev_assignment_experiments.tracking.transfer import file_digest


def build(source_bytes, destination):
    destination = Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=False)
    with zipfile.ZipFile(io.BytesIO(source_bytes)) as archive:
        env = json.loads(archive.read("environment.json"))
        for name in archive.namelist():
            path = Path(name)
            if path.is_absolute() or ".." in path.parts:
                raise ValueError("Source archive escape")
            if path.parts[0] not in {
                "ev_assignment",
                "ev_assignment_experiments",
                "environment.json",
            }:
                raise ValueError("Unexpected source member")
            target = destination / ("src/" + name if name != "environment.json" else name)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(archive.read(name))
    deps = [f"{k}=={env[k]}" for k in ("numpy", "scipy", "networkx", "psutil")]
    optional = (
        f'\n[project.optional-dependencies]\ngurobi = ["gurobipy=={env["gurobipy"]}"]\n'
        if env.get("gurobipy")
        else ""
    )
    text = (
        """[build-system]
requires = ["setuptools==75.8.0", "wheel==0.45.1"]
build-backend = "setuptools.build_meta"
[project]
name = "ev-assignment"
version = "0.2.0.dev0"
requires-python = ">=3.12,<3.13"
dependencies = """
        + json.dumps(deps)
        + """
[project.scripts]
ev-assignment = "ev_assignment_experiments.cli:main"
[tool.setuptools.packages.find]
where = ["src"]
"""
        + optional
    )
    (destination / "pyproject.toml").write_text(text)
    (destination / "requirements.lock").write_text("\n".join(deps) + "\n")
    (destination / "SOURCE.zip").write_bytes(source_bytes)
    files = {
        p.relative_to(destination).as_posix(): file_digest(p)
        for p in sorted(destination.rglob("*"))
        if p.is_file()
    }
    manifest = dict(
        schema_version=1,
        source_sha256=sha256(source_bytes),
        files=files,
        install="python3.12 -m pip install -r requirements.lock .",
        changes_active_release=False,
    )
    (destination / "RELEASE.json").write_bytes(encode(manifest))
    return manifest


def verify(destination):
    root = Path(destination).resolve()
    m = json.loads((root / "RELEASE.json").read_text())
    for name, h in m["files"].items():
        p = root / name
        if p.is_symlink() or not p.resolve().is_relative_to(root) or file_digest(p) != h:
            raise ValueError("Immutable release changed: " + name)
    raw = (root / "SOURCE.zip").read_bytes()
    if sha256(raw) != m["source_sha256"]:
        raise ValueError("Release source archive identity changed")
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        expected = set()
        for name in archive.namelist():
            member = Path(name)
            if (
                member.is_absolute()
                or ".." in member.parts
                or member.parts[0]
                not in {"ev_assignment", "ev_assignment_experiments", "environment.json"}
            ):
                raise ValueError("Source archive escapes release")
            relative = ("src/" + name) if name != "environment.json" else name
            expected.add(relative)
            target = root / relative
            if target.read_bytes() != archive.read(name):
                raise ValueError("Release differs from recorded source: " + name)
        actual = {p.relative_to(root).as_posix() for p in (root / "src").rglob("*.py")}
        if actual != {n for n in expected if n.endswith(".py")}:
            raise ValueError("Release has undeclared Python modules")
    return dict(passed=True, source_sha256=m["source_sha256"], files=len(m["files"]))

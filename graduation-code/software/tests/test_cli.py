from __future__ import annotations

import json

import pytest

from src.main import main, parse_args


def test_cli_defaults_to_factorization_and_solve_scope():
    # T12: the device executes the solve chain, so the CLI compiles the full
    # Ax=b artifact by default; --factorization-only keeps the old scope and
    # --emit-solve-commands stays as a compatibility alias.
    assert parse_args([]).emit_solve_commands is True
    assert parse_args(["--factorization-only"]).emit_solve_commands is False
    assert parse_args(["--emit-solve-commands"]).emit_solve_commands is True
    with pytest.raises(SystemExit):
        parse_args(["--factorization-only", "--emit-solve-commands"])


@pytest.mark.parametrize(
    ("extra", "scope"),
    [
        ([], "factorization_and_solve"),
        (["--emit-solve-commands"], "factorization_and_solve"),
        (["--factorization-only"], "factorization_only"),
    ],
)
def test_cli_writes_matching_operation_scope(tmp_path, capsys, extra, scope):
    out_dir = tmp_path / "artifact"
    main(["--n", "12", "--density", "0.2", "--out", str(out_dir), *extra])
    manifest = json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["compiler"]["operation_scope"] == scope
    assert f"operation_scope: {scope}" in capsys.readouterr().out

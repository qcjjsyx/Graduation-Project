# Shared FP64 front-tree policy fixture

`tree_policy_cases.json` pins the **same-policy** front-tree semantics — child
contribution assembly, pivot policy (`STATIC` / `DYNAMIC_PANEL`), threshold
handling, local P-vector direction and tree solve — so the two reference
implementations can be cross-validated against one committed expectation:

- Python host reference: `graduation-code/software/src/verify/front_reference.py`
  (consumed by `tests/test_reference_cross.py`);
- independent C++ reference: `graduation-code/systemc/reference/`
  (consumed by `fp64_reference_tests`, together with
  `../command_schema_v2/` and `../../systemc/reference/fixtures/`).

Both suites must reproduce the same `x_hat`, the same per-front local P-vectors
and the same pivot-failure kinds; that is the same-policy full-tree
cross-validation.  Node order in every case is children-before-parents.

`generate_fixture.py` regenerates the file deterministically from the Python
reference.  After regenerating, re-run **both** test suites; the C++ side must
keep passing unchanged, otherwise the two implementations diverged.

Schema: `fp64_reference_tree_v1`.  Per case:

| field | meaning |
| --- | --- |
| `policy` | `STATIC` or `DYNAMIC_PANEL` |
| `threshold_ratio` | binary32-normalized relative threshold |
| `parent` | node → parent, `-1` = root |
| `nodes[]` | `variables` (front row/col = global variable ids), `pivot_dim`, `front` (local contribution, update block zeroed, row-major) |
| `maps[]` | per node: child contributions `child`, `row_map`, `col_map` |
| `rhs` | RHS in the shared variable order |
| `expected` | `status: ok` with `x_hat` + per-node `p_local`, or `status: pivot_failure` with `kind`/`column`/`node_id` |

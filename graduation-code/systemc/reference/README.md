# FP64 mathematical reference

This directory implements the T02 `HOST_FP64_REFERENCE`. It is ordinary C++17
and deliberately does not include SystemC headers, call `sc_start`, read an
artifact memory image, or write golden values into a device buffer.

The public API covers dense partial-pivot Panel-LU, left/right TRSM,
`C - A*B` GEMM-Schur, complete dense solve, a small-front-tree solve helper,
and the three correctness metrics required by the verification plan.

Reference conventions:

- matrices are row-major FP64;
- panel pivot candidates are rows `k..m-1`;
- equal absolute values choose the lowest current logical row;
- `permutation[i]` means row `i` of `P*A` came from that original row;
- `minimum_pivot_ratio` is the minimum selected pivot magnitude divided by the
  maximum absolute input entry;
- `pivot_growth` is the largest absolute workspace value observed during
  elimination divided by the maximum absolute input entry;
- relative residual is `||Ax-b||₂ / (||A||F ||x||₂ + ||b||₂)`.

`solve_front_tree` supports the static child-to-parent fixtures used by T02. It
rejects delayed/cross-front pivots; those policies belong to T07 and later
device work.

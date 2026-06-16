# Multifrontal LU software pipeline

## Run

```powershell
python -m src.main --mtx path --out out_dir --seed 0 --ordering amd --effective-bits 27
```

If `--mtx` is omitted, a random SPD matrix is generated with `--n` and `--density`.

Ordering methods:
- `amd`: dependency-free minimum-degree ordering prototype
- `rcm`: reverse Cuthill-McKee fallback
- `identity`: no reordering

Quantization options:
- `--effective-bits`: mantissa magnitude limit uses `Q_use = 2^effective_bits - 1`
- `--clip-percentile`: default `100.0`, which follows the design note and uses the local maximum absolute value

## Outputs
- `tasks.bin`
- `map_table.bin`
- `front_q.bin` / `front_e.bin`
- `manifest.json`

## Quantization responsibility boundary

The quantization scheme is mainly a hardware-side design. The software pipeline
does not implement integer LU, TRSM, GEMM, final node-scale selection, or child
update numeric generation.

Software-side responsibility:
- run symbolic analysis and determine each node's `front_indices`
- extract each node's local original-matrix contribution `A_local`
- quantize each `A_local` into DDR input format `S_format`
- write int32 mantissas, int16 source exponents, node tasks, map tables, and
  manifest metadata for the hardware pipeline

Hardware-side responsibility:
- read software-prepared `A_local` sources and child update sources from DDR
- align sources by exponent during parent-node assembly
- accumulate the assembled front and determine final node-scale
- execute integer panel LU, TRSM, GEMM/Schur update
- write child update payloads for parent-node assembly

`front_q.bin` stores int32 mantissas for each node's local frontal contribution
`A_local`. `front_e.bin` stores one int16 source exponent per node. This is the
software-side `S_format` input used by the hardware assembly path; it is not a
single quantization pass over the original matrix.

`manifest.json` is validated after generation. The validator checks file sizes,
NodeTask record size, node/task counts, quantization metadata, memory alignment,
non-overlapping DDR regions, and map table decodability.

## Refactor layout
- `src/dataStruct.py`: Python-side data contract and binary ABI helpers
- `src/config.py`: pipeline, ordering, quantization, and memory configuration
- `src/matrix_io.py`: matrix loading and random SPD generation
- `src/symbolic/ordering.py`: ordering interface
- `src/symbolic/supernode.py`: structural supernode detection and front index generation
- `src/scheduler/map_gen.py`: child update variable to parent front mapping
- `src/quant/bfp_quant.py`: local contribution quantization and hardware-path reference checks
- `src/verify/manifest.py`: generated artifact consistency validation
- `src/pipeline.py`: end-to-end software-side generation flow

## Current symbolic assumptions
- Supernodes are merged from consecutive columns when their lower-column
  sparsity patterns match and they form an elimination-tree chain.
- Each node records `front_indices`; `NodeTask.pivot_dim` is the supernode size,
  while `NodeTask.total_dim` is the full front matrix dimension.
- Map table entries map a child's update variables into the parent front.

## Tests

```powershell
pytest
```

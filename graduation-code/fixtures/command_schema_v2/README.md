# Command schema v2 golden fixtures

These files are shared by the Python and C++ codec tests.  Schema **v2** is the
current mainline Command/Descriptor artifact format; it is unrelated to the
archived NodeTask ABI v2 (`archive/legacy-node-task-abi-v2/`).  The historical
Command/Descriptor v1 fixtures live in `../command_schema_v1/` and are kept
only for traceability and for negative tests (v2 must reject v1 records).

Every `.bin` is the byte-for-byte conversion of the same-name lowercase
hexadecimal `.hex` file:

```bash
xxd -r -p golden_command.hex golden_command.bin
```

`generate_fixtures.py` regenerates the whole directory deterministically.

## Records

| file | content |
| --- | --- |
| `golden_command` | one 32B Command record (bytes identical to the v1 golden: the command layout did not change) |
| `golden_descriptor` | one 64B `FRONT_DESC` with `pivot_policy = STATIC` and threshold bits `0x0` — the schema-v2 signature record |
| `static_front_descriptor` | `FRONT_DESC`, `pivot_policy = STATIC`, threshold ratio `0.0` |
| `dynamic_front_descriptor` | `FRONT_DESC`, `pivot_policy = DYNAMIC_PANEL`, threshold ratio `2^-12` (binary32 bits) |
| `golden_dependency_descriptor` | one `DEPENDENCY_DESC` for the token-payload tests (bytes identical to the v1 golden descriptor) |
| `golden_region_descriptor` | one `REGION_DESC` (8x8 FP32, 32B row stride, base `0x40`) |
| `golden_completion` | one 64B Completion record (bytes identical to the v1 golden) |
| `dependency_tokens` | token IDs 2, 4, 6 for the golden `DEPENDENCY_DESC` payload |
| `single_node_commands` | two-command stream (NODE_BEGIN, NODE_COMMIT) |
| `parent_child_commands` | three-command stream with a wait list (STORE_UPDATE, ASSEMBLE_EXTEND_ADD, NODE_COMMIT) |
| `token_states` | test-only u32 state vector `UNSIGNALED=0`, `READY=1`, `FAILED=2`; not a GCU scoreboard ABI |

## Schema v2 fields exercised here

`FRONT_DESC.body_words[8]` is the pivot policy
(`STATIC=0x01`, `DYNAMIC_PANEL=0x02`; `0x00` is invalid and marks a v1
record), `body_words[9]` is the relative pivot threshold as IEEE-754
binary32 bits (`[0, 1)`), scaled by `front_pre_elimination_max_abs` at
execution time.  The policy is never guessed from P-vector contents; the
P-vector region of an emitted image is initialized to the INT32 sentinel
`-1` and must be rewritten by `PANEL_LU` (`STATIC` writes the explicit
identity).

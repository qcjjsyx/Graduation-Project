# Command schema v1 golden fixtures

These files are shared by the Python and C++ codec tests. Every `.bin` is the
byte-for-byte conversion of the same-name lowercase hexadecimal `.hex` file.
The conversion command is:

```bash
xxd -r -p golden_command.hex golden_command.bin
```

`golden_command`, `golden_descriptor`, and `golden_completion` contain one
fixed-size record each. `single_node_commands` and `parent_child_commands`
exercise deterministic command streams without implementing an executor.

`token_states` is a test-only u32 little-endian state vector containing
`UNSIGNALED=0`, `READY=1`, and `FAILED=2`. It does not define the physical GCU
scoreboard layout. `dependency_tokens` contains token IDs 2, 4, and 6 for the
golden `DEPENDENCY_DESC` payload.

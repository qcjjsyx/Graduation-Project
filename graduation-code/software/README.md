# Multifrontal LU prototype

## Run

```powershell
python -m software.main --mtx path --out out_dir --seed 0
```

If `--mtx` is omitted, a random SPD matrix is generated with `--n` and `--density`.

## Outputs
- `tasks.bin`
- `map_table.bin`
- `front_q.bin` / `front_e.bin`
- `manifest.json`

## Tests

```powershell
pytest
```
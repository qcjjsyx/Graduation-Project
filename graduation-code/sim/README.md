python -m py_compile graduation-code/sim/hardware.py
python graduation-code/sim/hardware.py --n 32 --mode stable --seed 42 --ir-iters 5
python graduation-code/sim/hardware.py --n 32 --mode pivot_stress --seed 42 --ir-iters 5
python graduation-code/sim/hardware.py --n 256 --mode stable --seed 42 --ir-iters 5

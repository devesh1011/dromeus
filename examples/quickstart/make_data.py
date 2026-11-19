"""Write four nodes' private regression records for the quickstart.

Usage: python examples/quickstart/make_data.py OUTPUT_DIR

Each node gets its own NPZ (`data-0.npz` ... `data-3.npz`) of a different size,
drawn from the relation the example's public holdout measures:
target = 0.3 * first feature, plus a little noise.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


def main() -> None:
    output = Path(sys.argv[1])
    output.mkdir(parents=True, exist_ok=True)
    for node in range(4):
        rng = np.random.default_rng(100 + node)
        inputs = rng.normal(size=(64 + 16 * node, 3)).astype(np.float32)
        noise = rng.normal(scale=0.01, size=(len(inputs), 1))
        targets = (inputs[:, :1] * 0.3 + noise).astype(np.float32)
        np.savez(output / f"data-{node}.npz", inputs=inputs, targets=targets)
        print(f"wrote {output / f'data-{node}.npz'}: {len(inputs)} records")


if __name__ == "__main__":
    main()

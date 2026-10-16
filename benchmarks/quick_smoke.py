"""Quick smoke test: first 20 samples, 2 consistency repeats.

python benchmarks/quick_smoke.py --model ./models/model.gguf --quant Q4_K_M
"""

import sys

from pipette_adversarial.cli import main

if __name__ == "__main__":
    sys.exit(main(["--limit", "20", "--consistency-runs", "2", *sys.argv[1:]]))

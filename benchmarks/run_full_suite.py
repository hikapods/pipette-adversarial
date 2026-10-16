"""Full benchmark. Thin wrapper over the installed CLI.

    python benchmarks/run_full_suite.py --model ./models/lfm2-2.6b-Q4_K_M.gguf --quant Q4_K_M \
        --baseline-results results/lfm2-2.6b-F16-metal.json --output results/lfm2-2.6b-Q4_K_M-metal.json
"""

import sys

from pipette_adversarial.cli import main

if __name__ == "__main__":
    sys.exit(main())

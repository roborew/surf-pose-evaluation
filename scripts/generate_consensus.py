#!/usr/bin/env python3
"""
Standalone script for generating consensus pseudo-ground-truth.

⚠️  DEPRECATED: This script is no longer needed.
    Consensus generation is now integrated into run_evaluation.py

Use the main pipeline instead:
    python run_evaluation.py --run-name "my_experiment"

The pipeline automatically:
1. Selects validation clips (75 for Optuna, 200 for comparison)
2. Generates consensus GT on those clips
3. Runs Optuna with PCK validation
4. Runs comparison with the same consensus data

This file is kept for reference only and may be removed in future versions.
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def main():
    """Inform user to use the main pipeline."""
    print("\n" + "=" * 80)
    print("⚠️  DEPRECATED SCRIPT")
    print("=" * 80)
    print("\nThis standalone consensus generation script is deprecated.")
    print("Consensus generation is now integrated into the main pipeline.\n")
    print("Please use run_evaluation.py instead:")
    print("  python run_evaluation.py --run-name 'my_experiment'\n")
    print("The pipeline will automatically generate consensus and run validation.")
    print("=" * 80 + "\n")
    sys.exit(1)


if __name__ == "__main__":
    main()

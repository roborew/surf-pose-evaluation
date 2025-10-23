#!/usr/bin/env python3
"""
Quick test to verify consensus implementation is working.

Run this before full evaluation to check:
1. Imports work
2. ConsensusManager can be initialized
3. Quality filter can be created
"""

import sys
from pathlib import Path


def test_imports():
    """Test that consensus modules can be imported"""
    print("🔍 Testing imports...")
    try:
        from utils.consensus_manager import ConsensusManager

        print("   ✅ ConsensusManager imported")
    except ImportError as e:
        print(f"   ❌ Failed to import ConsensusManager: {e}")
        return False

    try:
        from utils.quality_filter import AdaptiveQualityFilter

        print("   ✅ AdaptiveQualityFilter imported")
    except ImportError as e:
        print(f"   ❌ Failed to import AdaptiveQualityFilter: {e}")
        return False

    try:
        from utils.consensus_generator import ConsensusGenerator

        print("   ✅ ConsensusGenerator imported")
    except ImportError as e:
        print(f"   ❌ Failed to import ConsensusGenerator: {e}")
        return False

    return True


def test_initialization():
    """Test that consensus components can be initialized"""
    print("\n🔧 Testing initialization...")

    try:
        from utils.consensus_manager import ConsensusManager
        from utils.quality_filter import AdaptiveQualityFilter
        import tempfile

        # Create quality filter
        quality_filter = AdaptiveQualityFilter(
            w_confidence=0.4, w_stability=0.4, w_completeness=0.2
        )
        print("   ✅ AdaptiveQualityFilter initialized")

        # Create temp cache directory
        with tempfile.TemporaryDirectory() as temp_dir:
            cache_dir = Path(temp_dir) / "consensus_cache"

            # Create consensus manager
            consensus_manager = ConsensusManager(
                consensus_models=["yolov8", "pytorch_pose", "mmpose"],
                quality_filter=quality_filter,
                cache_dir=cache_dir,
            )
            print("   ✅ ConsensusManager initialized")

            # Test leave-one-out logic
            for model in ["yolov8", "pytorch_pose", "mmpose", "mediapipe", "blazepose"]:
                models = consensus_manager.get_consensus_models_for_target(model)
                print(f"   ✅ {model} -> {models}")

        return True

    except Exception as e:
        print(f"   ❌ Initialization failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_config():
    """Test that consensus config is properly set"""
    print("\n⚙️  Testing configuration...")

    try:
        import yaml

        config_file = Path("configs/evaluation_config_production_optuna.yaml")
        if not config_file.exists():
            print(f"   ⚠️  Config file not found: {config_file}")
            return False

        with open(config_file, "r") as f:
            config = yaml.safe_load(f)

        use_consensus = config.get("optuna_validation", {}).get("use_consensus", False)

        if use_consensus:
            print(f"   ✅ use_consensus: {use_consensus}")

            models_to_optimize = config.get("models_to_optimize", [])
            print(f"   ✅ models_to_optimize: {models_to_optimize}")

            return True
        else:
            print(f"   ❌ use_consensus is False or missing!")
            return False

    except Exception as e:
        print(f"   ❌ Config test failed: {e}")
        return False


def main():
    """Run all tests"""
    print("=" * 60)
    print("CONSENSUS IMPLEMENTATION TEST")
    print("=" * 60)

    results = []

    # Test imports
    results.append(("Imports", test_imports()))

    # Test initialization
    results.append(("Initialization", test_initialization()))

    # Test config
    results.append(("Configuration", test_config()))

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    all_passed = True
    for test_name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{test_name}: {status}")
        if not passed:
            all_passed = False

    if all_passed:
        print("\n🎉 All tests passed! Consensus implementation is ready.")
        return 0
    else:
        print("\n⚠️  Some tests failed. Check errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())

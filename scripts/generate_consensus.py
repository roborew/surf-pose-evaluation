#!/usr/bin/env python3
"""
Standalone script for generating consensus pseudo-ground-truth.

Analyzes sessions, selects optimal consensus session, and generates
high-quality consensus annotations for both Optuna and comparison validation.

Usage:
    python scripts/generate_consensus.py --config configs/consensus_config.yaml
    python scripts/generate_consensus.py --session SESSION_070325 --force
"""

import argparse
import logging
import sys
import yaml
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from utils.session_analyzer import SessionAnalyzer
from utils.consensus_generator import ConsensusGenerator
from utils.quality_filter import AdaptiveQualityFilter
from utils.data_selection_manager import DataSelectionManager

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    """Load consensus configuration."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def analyze_and_select_session(config: dict, force_session: str = None) -> tuple:
    """
    Analyze sessions and select best for consensus.

    Returns:
        Tuple of (consensus_session, optuna_sessions, comparison_sessions)
    """
    logger.info("=" * 80)
    logger.info("STEP 1: Session Analysis")
    logger.info("=" * 80)

    # Initialize analyzer
    labels_path = (
        config.get("data_source", {}).get("annotations", {}).get("labels_path")
    )
    if not labels_path:
        # Fallback to default
        labels_path = (
            "./data/SD_02_SURF_FOOTAGE_PREPT/04_ANNOTATED/EXPORTED-MANEUVER-LABELS"
        )

    analyzer = SessionAnalyzer(labels_path)

    # Analyze all sessions
    sessions = analyzer.analyze_all_sessions()

    # Generate report
    report = analyzer.generate_report(sessions)
    logger.info("\n" + report)

    # Select consensus session
    if force_session:
        logger.info(f"Using forced consensus session: {force_session}")
        consensus_session = force_session

        if consensus_session not in sessions:
            raise ValueError(f"Session {force_session} not found in available sessions")
    else:
        consensus_session, stats = analyzer.recommend_consensus_session(sessions)
        logger.info(f"Selected consensus session: {consensus_session}")
        logger.info(f"  Maneuvers: {stats['total_maneuvers']}")
        logger.info(f"  Classes: {len(stats['maneuver_classes'])}")
        logger.info(f"  Variants: {', '.join(stats['variants'])}")

    # Split remaining sessions for Optuna and comparison
    optuna_ratio = config["consensus"]["data_splits"].get("optuna_ratio", 0.5)
    random_seed = config["consensus"]["data_splits"].get("random_seed", 42)

    optuna_sessions, comparison_sessions = analyzer.split_sessions_for_validation(
        sessions, consensus_session, optuna_ratio=optuna_ratio, random_seed=random_seed
    )

    return consensus_session, optuna_sessions, comparison_sessions


def setup_consensus_generator(config: dict) -> ConsensusGenerator:
    """Initialize consensus generator with quality filter."""
    logger.info("=" * 80)
    logger.info("STEP 2: Setup Consensus Generator")
    logger.info("=" * 80)

    # Initialize quality filter
    weights = config["consensus"]["quality_filter"]["composite_weights"]
    schedule = config["consensus"]["quality_filter"]["percentile_schedule"]

    quality_filter = AdaptiveQualityFilter(
        w_confidence=weights["confidence"],
        w_stability=weights["stability"],
        w_completeness=weights["completeness"],
        initialization_percentile=schedule["initialization"],
        growth_percentile=schedule["growth"],
        saturation_percentile=schedule["saturation"],
    )

    # Initialize consensus generator
    consensus_models = config["consensus"]["generation"]["consensus_models"]
    cache_path = config["consensus"]["generation"]["cache_path"]

    generator = ConsensusGenerator(
        consensus_models=consensus_models,
        quality_filter=quality_filter,
        cache_path=cache_path,
        config=config,
    )

    return generator


def load_clips_for_sessions(
    sessions: list, config: dict, max_clips: int = None
) -> list:
    """Load clips for specified sessions."""
    # This would integrate with DataSelectionManager
    # For now, we'll create a placeholder

    logger.info(f"Loading clips for {len(sessions)} sessions...")
    logger.info(f"Sessions: {sessions}")

    # TODO: Integrate with actual clip loading
    # This is a placeholder - actual implementation would use DataSelectionManager
    clips = []

    logger.warning("Clip loading not yet implemented - using placeholder")

    return clips


def generate_consensus_data(
    generator: ConsensusGenerator,
    consensus_session: str,
    optuna_sessions: list,
    comparison_sessions: list,
    config: dict,
):
    """Generate consensus for Optuna and comparison validation sets."""
    logger.info("=" * 80)
    logger.info("STEP 3: Generate Consensus Data")
    logger.info("=" * 80)

    # Load clips for each validation set
    optuna_clips = load_clips_for_sessions(
        optuna_sessions, config, max_clips=config["optuna_validation"].get("num_clips")
    )

    comparison_clips = load_clips_for_sessions(
        comparison_sessions,
        config,
        max_clips=config["comparison_validation"].get("num_clips"),
    )

    if not optuna_clips or not comparison_clips:
        logger.warning("No clips loaded - consensus generation skipped")
        logger.warning("This is expected if clip loading is not yet implemented")
        return

    # Generate consensus for both sets
    optuna_consensus, comparison_consensus = generator.generate_validation_sets(
        optuna_clips=optuna_clips,
        comparison_clips=comparison_clips,
        consensus_session=consensus_session,
    )

    logger.info("✅ Consensus generation complete!")
    logger.info(f"  Optuna consensus: {len(optuna_consensus.clips)} clips")
    logger.info(f"  Comparison consensus: {len(comparison_consensus.clips)} clips")


def update_config_with_sessions(
    config: dict,
    config_path: str,
    consensus_session: str,
    optuna_sessions: list,
    comparison_sessions: list,
):
    """Update config file with selected sessions."""
    logger.info("=" * 80)
    logger.info("STEP 4: Update Configuration")
    logger.info("=" * 80)

    # Update config
    config["consensus"]["generation"]["consensus_session"] = consensus_session
    config["consensus"]["data_splits"]["consensus_sessions"] = [consensus_session]
    config["consensus"]["data_splits"]["optuna_sessions"] = optuna_sessions
    config["consensus"]["data_splits"]["comparison_sessions"] = comparison_sessions

    # Save updated config
    with open(config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    logger.info(f"✅ Updated configuration saved to {config_path}")
    logger.info(f"  Consensus session: {consensus_session}")
    logger.info(f"  Optuna sessions: {len(optuna_sessions)}")
    logger.info(f"  Comparison sessions: {len(comparison_sessions)}")


def main():
    """Main consensus generation workflow."""
    parser = argparse.ArgumentParser(
        description="Generate consensus pseudo-ground-truth for pose validation"
    )

    parser.add_argument(
        "--config",
        type=str,
        default="configs/consensus_config.yaml",
        help="Path to consensus configuration file",
    )

    parser.add_argument(
        "--session",
        type=str,
        default=None,
        help="Force specific session for consensus (e.g., SESSION_070325)",
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help="Force regeneration even if consensus data exists",
    )

    parser.add_argument(
        "--analyze-only",
        action="store_true",
        help="Only analyze sessions without generating consensus",
    )

    args = parser.parse_args()

    try:
        # Load configuration
        logger.info(f"Loading configuration from {args.config}")
        config = load_config(args.config)

        # Step 1: Analyze and select sessions
        consensus_session, optuna_sessions, comparison_sessions = (
            analyze_and_select_session(config, force_session=args.session)
        )

        # Step 4: Update config with selected sessions
        update_config_with_sessions(
            config, args.config, consensus_session, optuna_sessions, comparison_sessions
        )

        if args.analyze_only:
            logger.info("Analysis complete (--analyze-only mode)")
            return

        # Check if consensus data already exists
        cache_path = Path(config["consensus"]["generation"]["cache_path"])
        optuna_path = cache_path / "optuna_validation"
        comparison_path = cache_path / "comparison_test"

        if not args.force and optuna_path.exists() and comparison_path.exists():
            logger.info("✅ Consensus data already exists")
            logger.info("   Use --force to regenerate")
            return

        # Step 2: Setup generator
        generator = setup_consensus_generator(config)

        # Step 3: Generate consensus data
        generate_consensus_data(
            generator, consensus_session, optuna_sessions, comparison_sessions, config
        )

        logger.info("=" * 80)
        logger.info("✅ CONSENSUS GENERATION COMPLETE")
        logger.info("=" * 80)
        logger.info("Next steps:")
        logger.info("  1. Review quality reports in results/consensus_quality_reports/")
        logger.info(
            "  2. Run Optuna optimization: python run_evaluation.py --run-optuna"
        )
        logger.info("  3. Run comparison: python run_evaluation.py --run-comparison")

    except Exception as e:
        logger.error(f"❌ Consensus generation failed: {e}")
        raise


if __name__ == "__main__":
    main()

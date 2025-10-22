"""
Session analyzer for identifying optimal consensus generation session.

Analyzes all available sessions to recommend the best SONY_300 session for
consensus pseudo-ground-truth generation based on maneuver count, diversity,
and quality metrics.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict, Counter
import numpy as np

logger = logging.getLogger(__name__)


class SessionAnalyzer:
    """Analyzes surf footage sessions for consensus generation selection."""

    def __init__(self, labels_base_path: str):
        """
        Initialize session analyzer.

        Args:
            labels_base_path: Base path to annotation labels directory
        """
        self.labels_base_path = Path(labels_base_path)
        self.sony_300_path = self.labels_base_path / "sony_300"
        self.sony_70_path = self.labels_base_path / "sony_70"

    def analyze_all_sessions(self) -> Dict[str, Dict]:
        """
        Analyze all sessions from both SONY_300 and SONY_70.

        Returns:
            Dictionary mapping session names to analysis results
        """
        logger.info("Analyzing all available sessions...")

        all_sessions = {}

        # Analyze SONY_300 sessions
        if self.sony_300_path.exists():
            sony_300_sessions = self._analyze_camera_sessions(
                self.sony_300_path, camera_type="SONY_300"
            )
            all_sessions.update(sony_300_sessions)

        # Analyze SONY_70 sessions
        if self.sony_70_path.exists():
            sony_70_sessions = self._analyze_camera_sessions(
                self.sony_70_path, camera_type="SONY_70"
            )
            all_sessions.update(sony_70_sessions)

        return all_sessions

    def _analyze_camera_sessions(
        self, camera_path: Path, camera_type: str
    ) -> Dict[str, Dict]:
        """Analyze sessions from a specific camera."""
        sessions = defaultdict(
            lambda: {
                "camera": camera_type,
                "variants": [],
                "total_clips": 0,
                "total_maneuvers": 0,
                "maneuver_classes": Counter(),
                "execution_scores": [],
                "session_files": [],
            }
        )

        for json_file in camera_path.glob("*.json"):
            if json_file.name.startswith("."):
                continue

            session_name = self._extract_session_name(json_file.name, camera_type)
            variant = self._extract_variant(json_file.name)

            # Parse annotation file
            stats = self._parse_annotation_file(json_file)

            # Aggregate session statistics
            sessions[session_name]["variants"].append(variant)
            sessions[session_name]["total_clips"] += stats["clip_count"]
            sessions[session_name]["total_maneuvers"] += stats["maneuver_count"]
            sessions[session_name]["maneuver_classes"].update(stats["maneuver_classes"])
            sessions[session_name]["execution_scores"].extend(stats["execution_scores"])
            sessions[session_name]["session_files"].append(json_file.name)

        return dict(sessions)

    def _extract_session_name(self, filename: str, camera_type: str) -> str:
        """
        Extract base session name from filename.

        Examples:
            SONY_300_SESSION_070325_FULL.json -> SESSION_070325
            SONY_70_SESSION_020325_WIDE.json -> SESSION_020325
        """
        # Remove camera prefix
        name = filename.replace(f"{camera_type}_", "")

        # Extract session date (format: SESSION_DDMMYY)
        parts = name.split("_")
        if len(parts) >= 2:
            return f"{parts[0]}_{parts[1]}"

        return name.replace(".json", "")

    def _extract_variant(self, filename: str) -> str:
        """
        Extract zoom variant from filename.

        Returns: 'FULL', 'WIDE', or 'STANDARD'
        """
        if "_FULL" in filename:
            return "FULL"
        elif "_WIDE" in filename:
            return "WIDE"
        else:
            return "STANDARD"

    def _parse_annotation_file(self, json_path: Path) -> Dict:
        """Parse annotation file and extract statistics."""
        try:
            with open(json_path, "r") as f:
                data = json.load(f)
        except Exception as e:
            logger.warning(f"Failed to parse {json_path}: {e}")
            return {
                "clip_count": 0,
                "maneuver_count": 0,
                "maneuver_classes": Counter(),
                "execution_scores": [],
            }

        clip_count = len(data)
        maneuver_count = 0
        maneuver_classes = Counter()
        execution_scores = []

        for clip in data:
            if "annotations" not in clip or not clip["annotations"]:
                continue

            for annotation in clip["annotations"]:
                if "result" not in annotation:
                    continue

                for result in annotation["result"]:
                    if result.get("type") == "labels":
                        maneuver_count += 1
                        labels = result.get("value", {}).get("labels", [])
                        maneuver_classes.update(labels)

                    elif result.get("type") == "choices":
                        # Extract execution scores
                        choices = result.get("value", {}).get("choices", [])
                        for choice in choices:
                            try:
                                score = int(choice)
                                execution_scores.append(score)
                            except (ValueError, TypeError):
                                pass

        return {
            "clip_count": clip_count,
            "maneuver_count": maneuver_count,
            "maneuver_classes": maneuver_classes,
            "execution_scores": execution_scores,
        }

    def recommend_consensus_session(
        self,
        sessions: Dict[str, Dict],
        prefer_camera: str = "SONY_300",
        min_maneuvers: int = 100,
        min_diversity: int = 5,
    ) -> Tuple[str, Dict]:
        """
        Recommend best session for consensus generation.

        Args:
            sessions: Session analysis results
            prefer_camera: Preferred camera type
            min_maneuvers: Minimum number of maneuvers required
            min_diversity: Minimum number of unique maneuver classes

        Returns:
            Tuple of (session_name, session_stats)
        """
        logger.info("Recommending consensus session...")

        # Filter sessions by camera preference and requirements
        candidates = []

        for session_name, stats in sessions.items():
            if stats["camera"] != prefer_camera:
                continue

            if stats["total_maneuvers"] < min_maneuvers:
                logger.debug(
                    f"Skipping {session_name}: only {stats['total_maneuvers']} maneuvers"
                )
                continue

            num_unique_classes = len(stats["maneuver_classes"])
            if num_unique_classes < min_diversity:
                logger.debug(
                    f"Skipping {session_name}: only {num_unique_classes} maneuver types"
                )
                continue

            # Calculate quality score
            quality_score = self._calculate_session_quality(stats)

            candidates.append((session_name, stats, quality_score))

        if not candidates:
            logger.warning(
                "No sessions meet minimum requirements, relaxing constraints..."
            )
            # Fallback: just pick session with most maneuvers
            candidates = [
                (name, stats, stats["total_maneuvers"])
                for name, stats in sessions.items()
                if stats["camera"] == prefer_camera
            ]

        # Sort by quality score
        candidates.sort(key=lambda x: x[2], reverse=True)

        if candidates:
            best_session, best_stats, best_score = candidates[0]
            logger.info(
                f"Recommended session: {best_session} "
                f"(score: {best_score:.2f}, maneuvers: {best_stats['total_maneuvers']})"
            )
            return best_session, best_stats

        raise ValueError("No suitable consensus session found")

    def _calculate_session_quality(self, stats: Dict) -> float:
        """
        Calculate quality score for session selection.

        Considers:
        - Number of maneuvers (more is better)
        - Maneuver diversity (more classes is better)
        - Average execution score (higher is better)
        - Number of variants (having FULL/WIDE is better)
        """
        # Maneuver count (normalized to 0-100 range, assuming 200 is excellent)
        maneuver_score = min(stats["total_maneuvers"] / 2.0, 100.0)

        # Diversity score (normalized, assuming 10+ classes is excellent)
        diversity_score = min(len(stats["maneuver_classes"]) * 10.0, 100.0)

        # Execution quality (0-10 scale, convert to 0-100)
        if stats["execution_scores"]:
            avg_execution = np.mean(stats["execution_scores"])
            execution_score = avg_execution * 10.0
        else:
            execution_score = 50.0  # Neutral if no scores

        # Variant completeness (bonus for having multiple zoom variants)
        variant_score = len(stats["variants"]) * 10.0

        # Weighted combination
        quality = (
            0.4 * maneuver_score
            + 0.3 * diversity_score
            + 0.2 * execution_score
            + 0.1 * variant_score
        )

        return quality

    def generate_report(self, sessions: Dict[str, Dict]) -> str:
        """Generate human-readable analysis report."""
        report_lines = ["=" * 80, "SESSION ANALYSIS REPORT", "=" * 80, ""]

        # Sort sessions by total maneuvers
        sorted_sessions = sorted(
            sessions.items(), key=lambda x: x[1]["total_maneuvers"], reverse=True
        )

        for session_name, stats in sorted_sessions:
            report_lines.append(f"Session: {session_name}")
            report_lines.append(f"  Camera: {stats['camera']}")
            report_lines.append(f"  Variants: {', '.join(stats['variants'])}")
            report_lines.append(f"  Total Clips: {stats['total_clips']}")
            report_lines.append(f"  Total Maneuvers: {stats['total_maneuvers']}")
            report_lines.append(f"  Maneuver Classes: {len(stats['maneuver_classes'])}")

            if stats["execution_scores"]:
                avg_score = np.mean(stats["execution_scores"])
                report_lines.append(f"  Avg Execution Score: {avg_score:.2f}")

            # Top maneuver types
            top_maneuvers = stats["maneuver_classes"].most_common(5)
            report_lines.append("  Top Maneuvers:")
            for maneuver, count in top_maneuvers:
                report_lines.append(f"    - {maneuver}: {count}")

            report_lines.append("")

        report_lines.append("=" * 80)

        return "\n".join(report_lines)

    def split_sessions_for_validation(
        self,
        sessions: Dict[str, Dict],
        consensus_session: str,
        optuna_ratio: float = 0.5,
        random_seed: int = 42,
    ) -> Tuple[List[str], List[str]]:
        """
        Split non-consensus sessions into Optuna and comparison sets.

        Args:
            sessions: All session analysis results
            consensus_session: Session reserved for consensus generation
            optuna_ratio: Fraction of remaining sessions for Optuna validation
            random_seed: Random seed for reproducibility

        Returns:
            Tuple of (optuna_sessions, comparison_sessions)
        """
        # Get all sessions except consensus
        available_sessions = [
            name for name in sessions.keys() if name != consensus_session
        ]

        # Shuffle with fixed seed
        np.random.seed(random_seed)
        np.random.shuffle(available_sessions)

        # Split by ratio
        split_idx = int(len(available_sessions) * optuna_ratio)
        optuna_sessions = available_sessions[:split_idx]
        comparison_sessions = available_sessions[split_idx:]

        logger.info(f"Session split:")
        logger.info(f"  Consensus: {consensus_session}")
        logger.info(f"  Optuna: {len(optuna_sessions)} sessions")
        logger.info(f"  Comparison: {len(comparison_sessions)} sessions")

        return optuna_sessions, comparison_sessions


def main():
    """Standalone session analysis script."""
    import argparse

    parser = argparse.ArgumentParser(description="Analyze surf footage sessions")
    parser.add_argument(
        "--labels-path",
        type=str,
        default="./data/SD_02_SURF_FOOTAGE_PREPT/04_ANNOTATED/EXPORTED-MANEUVER-LABELS",
        help="Path to annotation labels directory",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./session_analysis_report.txt",
        help="Output path for analysis report",
    )

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Run analysis
    analyzer = SessionAnalyzer(args.labels_path)
    sessions = analyzer.analyze_all_sessions()

    # Generate report
    report = analyzer.generate_report(sessions)
    print(report)

    # Save to file
    with open(args.output, "w") as f:
        f.write(report)

    # Recommend consensus session
    try:
        consensus_session, stats = analyzer.recommend_consensus_session(sessions)
        print(f"\n🎯 Recommended consensus session: {consensus_session}")
        print(f"   - Maneuvers: {stats['total_maneuvers']}")
        print(f"   - Classes: {len(stats['maneuver_classes'])}")
        print(f"   - Variants: {', '.join(stats['variants'])}")
    except ValueError as e:
        print(f"\n❌ Error: {e}")


if __name__ == "__main__":
    main()

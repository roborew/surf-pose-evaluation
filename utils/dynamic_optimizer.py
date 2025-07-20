#!/usr/bin/env python3
"""
Dynamic Optimizer for Intelligent Time Allocation
Manages optimization time allocation across models based on complexity and performance
"""

import logging
import time
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from pathlib import Path
import json

logger = logging.getLogger(__name__)


@dataclass
class ModelOptimizationResult:
    """Results from model optimization"""

    model_name: str
    trials_completed: int
    best_score: float
    time_taken: float
    early_stopped: bool
    efficiency: float = 0.0  # best_score / time_taken

    def __post_init__(self):
        if self.time_taken > 0:
            self.efficiency = self.best_score / self.time_taken


class DynamicOptimizer:
    """Manages dynamic time allocation across models"""

    def __init__(self, total_time_hours: float = 24, config: Dict = None):
        self.total_time_seconds = total_time_hours * 3600
        self.model_results: Dict[str, ModelOptimizationResult] = {}
        self.remaining_time = self.total_time_seconds
        self.config = config or {}

        # Model complexity scores (higher = more time needed)
        self.model_complexity = {
            "mediapipe": 0.15,  # Fastest, least complex
            "yolov8_pose": 0.20,  # Fast, moderate complexity
            "blazepose": 0.20,  # Moderate speed, moderate complexity
            "pytorch_pose": 0.25,  # Slower, higher complexity
            "mmpose": 0.30,  # Slowest, most complex
        }

        # Initial time allocation based on complexity
        self.initial_allocation = self._calculate_initial_allocation()

        logger.info(
            f"Dynamic optimizer initialized with {total_time_hours}h total time"
        )
        logger.info(
            f"Initial allocation: {self._format_allocation(self.initial_allocation)}"
        )

    def _calculate_initial_allocation(self) -> Dict[str, float]:
        """Calculate initial time allocation based on model complexity"""
        total_complexity = sum(self.model_complexity.values())
        allocation = {}

        for model_name, complexity in self.model_complexity.items():
            allocation[model_name] = (
                complexity / total_complexity
            ) * self.total_time_seconds

        return allocation

    def _format_allocation(self, allocation: Dict[str, float]) -> str:
        """Format allocation for logging"""
        return ", ".join(
            [f"{model}: {time/3600:.1f}h" for model, time in allocation.items()]
        )

    def get_time_for_model(self, model_name: str) -> float:
        """Get allocated time for a specific model"""
        if model_name in self.initial_allocation:
            return self.initial_allocation[model_name]
        else:
            # Default allocation for unknown models
            return self.total_time_seconds / len(self.model_complexity)

    def update_model_result(
        self,
        model_name: str,
        trials_completed: int,
        best_score: float,
        time_taken: float,
        early_stopped: bool = False,
    ):
        """Update results and adjust remaining allocations"""
        result = ModelOptimizationResult(
            model_name=model_name,
            trials_completed=trials_completed,
            best_score=best_score,
            time_taken=time_taken,
            early_stopped=early_stopped,
        )

        self.model_results[model_name] = result
        self.remaining_time -= time_taken

        logger.info(
            f"Model {model_name} completed: {trials_completed} trials, "
            f"score: {best_score:.4f}, time: {time_taken/60:.1f}min, "
            f"efficiency: {result.efficiency:.4f}"
        )

        # Redistribute remaining time to unfinished models
        self._redistribute_time()

    def _redistribute_time(self):
        """Redistribute remaining time to models that need it"""
        unfinished_models = [
            m for m in self.model_complexity.keys() if m not in self.model_results
        ]

        if unfinished_models and self.remaining_time > 0:
            # Calculate new allocation based on remaining complexity
            remaining_complexity = sum(
                self.model_complexity[m] for m in unfinished_models
            )

            if remaining_complexity > 0:
                for model_name in unfinished_models:
                    complexity_ratio = (
                        self.model_complexity[model_name] / remaining_complexity
                    )
                    self.initial_allocation[model_name] = (
                        self.remaining_time * complexity_ratio
                    )

                logger.info(
                    f"Redistributed {self.remaining_time/3600:.1f}h among {len(unfinished_models)} models"
                )
                logger.info(
                    f"New allocation: {self._format_allocation(self.initial_allocation)}"
                )

    def get_optimization_summary(self) -> Dict[str, Any]:
        """Get summary of optimization results"""
        if not self.model_results:
            return {"status": "No models completed"}

        total_trials = sum(r.trials_completed for r in self.model_results.values())
        total_time = sum(r.time_taken for r in self.model_results.values())
        avg_score = sum(r.best_score for r in self.model_results.values()) / len(
            self.model_results
        )

        # Find best performing model
        best_model = max(self.model_results.values(), key=lambda x: x.best_score)
        most_efficient = max(self.model_results.values(), key=lambda x: x.efficiency)

        summary = {
            "total_models_completed": len(self.model_results),
            "total_trials": total_trials,
            "total_time_hours": total_time / 3600,
            "remaining_time_hours": self.remaining_time / 3600,
            "average_score": avg_score,
            "best_model": {
                "name": best_model.model_name,
                "score": best_model.best_score,
                "trials": best_model.trials_completed,
            },
            "most_efficient": {
                "name": most_efficient.model_name,
                "efficiency": most_efficient.efficiency,
                "score": most_efficient.best_score,
            },
            "early_stopped_models": [
                r.model_name for r in self.model_results.values() if r.early_stopped
            ],
            "model_results": {
                name: {
                    "trials": result.trials_completed,
                    "score": result.best_score,
                    "time_hours": result.time_taken / 3600,
                    "efficiency": result.efficiency,
                    "early_stopped": result.early_stopped,
                }
                for name, result in self.model_results.items()
            },
        }

        return summary

    def save_summary(self, output_path: Path):
        """Save optimization summary to file"""
        summary = self.get_optimization_summary()

        try:
            with open(output_path, "w") as f:
                json.dump(summary, f, indent=2)
            logger.info(f"Optimization summary saved to {output_path}")
        except Exception as e:
            logger.error(f"Failed to save optimization summary: {e}")

    def print_summary(self):
        """Print optimization summary to console"""
        summary = self.get_optimization_summary()

        if summary.get("status") == "No models completed":
            print("No models have completed optimization yet.")
            return

        print("\n" + "=" * 60)
        print("DYNAMIC OPTIMIZATION SUMMARY")
        print("=" * 60)
        print(f"Models completed: {summary['total_models_completed']}")
        print(f"Total trials: {summary['total_trials']}")
        print(f"Total time: {summary['total_time_hours']:.1f}h")
        print(f"Remaining time: {summary['remaining_time_hours']:.1f}h")
        print(f"Average score: {summary['average_score']:.4f}")

        print(f"\nBest performing model: {summary['best_model']['name']}")
        print(f"  Score: {summary['best_model']['score']:.4f}")
        print(f"  Trials: {summary['best_model']['trials']}")

        print(f"\nMost efficient model: {summary['most_efficient']['name']}")
        print(f"  Efficiency: {summary['most_efficient']['efficiency']:.4f}")
        print(f"  Score: {summary['most_efficient']['score']:.4f}")

        if summary["early_stopped_models"]:
            print(
                f"\nEarly stopped models: {', '.join(summary['early_stopped_models'])}"
            )

        print("\nDetailed Results:")
        print("-" * 40)
        for model_name, result in summary["model_results"].items():
            status = "🛑" if result["early_stopped"] else "✅"
            print(
                f"{status} {model_name}: {result['trials']} trials, "
                f"score: {result['score']:.4f}, time: {result['time_hours']:.1f}h, "
                f"efficiency: {result['efficiency']:.4f}"
            )

        print("=" * 60)


def create_dynamic_optimizer_from_config(config: Dict) -> DynamicOptimizer:
    """Create dynamic optimizer from configuration"""
    optuna_config = config.get("optuna", {})
    total_time_hours = optuna_config.get("total_time_hours", 24)

    return DynamicOptimizer(total_time_hours=total_time_hours, config=config)

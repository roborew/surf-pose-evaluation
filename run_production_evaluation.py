#!/usr/bin/env python3
"""
Production Evaluation Pipeline
Automated two-phase evaluation: Optuna optimization + Model comparison
"""

import os
import sys
import json
import yaml
import argparse
import subprocess
from pathlib import Path
from datetime import datetime
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ProductionEvaluationPipeline:
    """Automated pipeline for production pose model evaluation"""
    
    def __init__(self, models=None, max_clips=None, skip_optuna=False):
        self.models = models or ["mediapipe", "blazepose", "yolov8_pose", "pytorch_pose"]
        self.max_clips = max_clips
        self.skip_optuna = skip_optuna
        self.results_dir = Path("./results")
        self.best_params_dir = self.results_dir / "best_params"
        
        # Ensure directories exist
        self.results_dir.mkdir(exist_ok=True)
        self.best_params_dir.mkdir(exist_ok=True)
    
    def run_optuna_optimization(self):
        """Phase 1: Run Optuna hyperparameter optimization"""
        logger.info("🔍 PHASE 1: Starting Optuna hyperparameter optimization")
        
        cmd = [
            "python", "evaluate_pose_models.py",
            "--config", "configs/evaluation_config_production_optuna.yaml",
            "--use-optuna",
            "--models"
        ] + self.models
        
        if self.max_clips:
            cmd.extend(["--max-clips", str(self.max_clips)])
        
        logger.info(f"Running command: {' '.join(cmd)}")
        
        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            logger.info("✅ Optuna optimization completed successfully")
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"❌ Optuna optimization failed: {e}")
            logger.error(f"Stdout: {e.stdout}")
            logger.error(f"Stderr: {e.stderr}")
            return False
    
    def extract_best_parameters(self):
        """Extract best parameters from Optuna MLflow runs"""
        logger.info("📊 Extracting best parameters from Optuna results")
        
        try:
            import mlflow
            
            # Connect to MLflow
            mlflow.set_tracking_uri("./results/mlruns")
            
            # Get the Optuna experiment
            experiment = mlflow.get_experiment_by_name("surf_pose_production_optuna")
            if not experiment:
                logger.error("❌ Optuna experiment not found")
                return False
            
            best_params = {}
            
            # For each model, find the best_full_eval run
            for model in self.models:
                runs = mlflow.search_runs(
                    experiment_ids=[experiment.experiment_id],
                    filter_string=f"tags.mlflow.runName LIKE '{model}_optuna_best_full_eval'",
                    max_results=1
                )
                
                if runs.empty:
                    logger.warning(f"⚠️ No best full eval run found for {model}")
                    continue
                
                run = runs.iloc[0]
                model_params = {}
                
                # Extract parameters (filter out non-hyperparameters)
                for param_name, param_value in run.params.items():
                    if not param_name.startswith(('model_name', 'optimization_mode', 'data_scope', 'purpose')):
                        model_params[param_name] = param_value
                
                best_params[model] = model_params
                logger.info(f"✅ Extracted best parameters for {model}: {model_params}")
            
            # Save best parameters
            best_params_file = self.best_params_dir / "best_parameters.yaml"
            with open(best_params_file, 'w') as f:
                yaml.dump(best_params, f, default_flow_style=False)
            
            logger.info(f"💾 Best parameters saved to {best_params_file}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to extract best parameters: {e}")
            return False
    
    def run_model_comparison(self):
        """Phase 2: Run model comparison with best parameters"""
        logger.info("🏆 PHASE 2: Starting model comparison with optimal parameters")
        
        cmd = [
            "python", "evaluate_pose_models.py",
            "--config", "configs/evaluation_config_production_comparison.yaml",
            "--models"
        ] + self.models
        
        if self.max_clips:
            cmd.extend(["--max-clips", str(self.max_clips)])
        
        logger.info(f"Running command: {' '.join(cmd)}")
        
        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            logger.info("✅ Model comparison completed successfully")
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"❌ Model comparison failed: {e}")
            logger.error(f"Stdout: {e.stdout}")
            logger.error(f"Stderr: {e.stderr}")
            return False
    
    def generate_summary_report(self):
        """Generate a summary report comparing the results"""
        logger.info("📋 Generating summary report")
        
        try:
            import mlflow
            import pandas as pd
            
            mlflow.set_tracking_uri("./results/mlruns")
            
            # Get comparison experiment
            experiment = mlflow.get_experiment_by_name("surf_pose_production_comparison")
            if not experiment:
                logger.error("❌ Comparison experiment not found")
                return False
            
            # Get all runs from comparison
            runs = mlflow.search_runs(
                experiment_ids=[experiment.experiment_id],
                order_by=["metrics.pose_pck_error_mean ASC"]  # Best accuracy first
            )
            
            if runs.empty:
                logger.error("❌ No comparison runs found")
                return False
            
            # Create summary
            summary = {
                "evaluation_date": datetime.now().isoformat(),
                "models_evaluated": self.models,
                "dataset_size": self.max_clips or "full",
                "results": []
            }
            
            for _, run in runs.iterrows():
                model_result = {
                    "model": run.get('params.model_name', 'unknown'),
                    "run_name": run.get('tags.mlflow.runName', 'unknown'),
                    "accuracy": {
                        "pck_error_mean": run.get('metrics.pose_pck_error_mean', None),
                        "detection_f1": run.get('metrics.pose_detection_f1_mean', None)
                    },
                    "performance": {
                        "fps_mean": run.get('metrics.perf_fps_mean', None),
                        "inference_time_ms": run.get('metrics.perf_avg_inference_time_mean', None),
                        "memory_usage_gb": run.get('metrics.perf_max_memory_usage_mean', None)
                    }
                }
                summary["results"].append(model_result)
            
            # Save summary
            summary_file = self.results_dir / "production_evaluation_summary.json"
            with open(summary_file, 'w') as f:
                json.dump(summary, f, indent=2)
            
            logger.info(f"📊 Summary report saved to {summary_file}")
            
            # Print summary to console
            print("\n" + "="*60)
            print("🏆 PRODUCTION EVALUATION SUMMARY")
            print("="*60)
            
            for result in summary["results"]:
                print(f"\n📍 {result['model'].upper()}")
                print(f"   • Accuracy (PCK Error): {result['accuracy']['pck_error_mean']:.4f}")
                print(f"   • Detection F1: {result['accuracy']['detection_f1']:.4f}")
                print(f"   • Speed (FPS): {result['performance']['fps_mean']:.2f}")
                print(f"   • Inference Time: {result['performance']['inference_time_ms']:.2f}ms")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to generate summary report: {e}")
            return False
    
    def run_full_pipeline(self):
        """Run the complete two-phase evaluation pipeline"""
        logger.info("🚀 Starting Production Evaluation Pipeline")
        logger.info(f"   • Models: {', '.join(self.models)}")
        logger.info(f"   • Max clips: {self.max_clips or 'full dataset'}")
        logger.info(f"   • Skip Optuna: {self.skip_optuna}")
        
        success = True
        
        # Phase 1: Optuna optimization
        if not self.skip_optuna:
            if not self.run_optuna_optimization():
                logger.error("❌ Pipeline failed at Optuna optimization phase")
                return False
            
            if not self.extract_best_parameters():
                logger.error("❌ Pipeline failed at parameter extraction phase")
                return False
        else:
            logger.info("⏭️ Skipping Optuna optimization (using existing parameters)")
        
        # Phase 2: Model comparison
        if not self.run_model_comparison():
            logger.error("❌ Pipeline failed at model comparison phase")
            return False
        
        # Generate summary
        if not self.generate_summary_report():
            logger.error("❌ Pipeline failed at summary generation phase")
            return False
        
        logger.info("🎉 Production evaluation pipeline completed successfully!")
        logger.info("📊 Check MLflow UI for detailed results:")
        logger.info("   • Optuna results: experiment 'surf_pose_production_optuna'")
        logger.info("   • Comparison results: experiment 'surf_pose_production_comparison'")
        logger.info("   • Summary: ./results/production_evaluation_summary.json")
        
        return True

def main():
    parser = argparse.ArgumentParser(
        description="Production Evaluation Pipeline - Automated Optuna + Model Comparison"
    )
    parser.add_argument(
        "--models", 
        nargs="+", 
        default=["mediapipe", "blazepose", "yolov8_pose", "pytorch_pose"],
        help="Models to evaluate"
    )
    parser.add_argument(
        "--max-clips", 
        type=int, 
        help="Maximum number of clips to process (default: full dataset)"
    )
    parser.add_argument(
        "--skip-optuna", 
        action="store_true",
        help="Skip Optuna optimization and use existing best parameters"
    )
    parser.add_argument(
        "--optuna-only", 
        action="store_true",
        help="Run only Optuna optimization phase"
    )
    parser.add_argument(
        "--comparison-only", 
        action="store_true",
        help="Run only model comparison phase (requires existing best parameters)"
    )
    
    args = parser.parse_args()
    
    # Create pipeline
    pipeline = ProductionEvaluationPipeline(
        models=args.models,
        max_clips=args.max_clips,
        skip_optuna=args.skip_optuna
    )
    
    # Run requested phases
    if args.optuna_only:
        success = pipeline.run_optuna_optimization() and pipeline.extract_best_parameters()
    elif args.comparison_only:
        success = pipeline.run_model_comparison() and pipeline.generate_summary_report()
    else:
        success = pipeline.run_full_pipeline()
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main() 
#!/usr/bin/env python3
"""
Test Data Splits Generation
Verify that data splits are generated and saved correctly in run folders
"""

import yaml
import logging
from pathlib import Path
from utils.run_manager import RunManager

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_data_splits_generation():
    """Test the new data splits generation functionality"""
    
    print("🧪 Testing Data Splits Generation")
    print("=" * 50)
    
    # Load config
    config_path = "configs/evaluation_config_production_optuna_enhanced.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    print(f"📋 Loaded config: {config_path}")
    
    # Create run manager
    run_manager = RunManager(run_name="test_data_splits")
    print(f"🏗️ Created run: {run_manager.run_dir}")
    
    # Generate data splits
    print("\n📊 Generating data splits...")
    split_files = run_manager.generate_data_splits(config)
    
    # Verify files were created
    print("\n✅ Verification:")
    for split_name, split_path in split_files.items():
        split_file = Path(split_path)
        if split_file.exists():
            print(f"   ✅ {split_name}: {split_file.name}")
        else:
            print(f"   ❌ {split_name}: {split_file.name} - NOT FOUND")
    
    # Check metadata file
    metadata_file = run_manager.data_splits_dir / "splits_metadata.json"
    if metadata_file.exists():
        print(f"   ✅ Metadata: splits_metadata.json")
    else:
        print(f"   ❌ Metadata: splits_metadata.json - NOT FOUND")
    
    # Show directory structure
    print(f"\n📁 Run directory structure:")
    print(f"   {run_manager.run_dir}")
    print(f"   ├── data_selections/")
    print(f"   │   ├── optuna_selection.json")
    print(f"   │   ├── comparison_selection.json")
    print(f"   │   └── visualization_selection.json")
    print(f"   └── data_splits/")
    print(f"       ├── train_split.json")
    print(f"       ├── val_split.json")
    print(f"       ├── test_split.json")
    print(f"       └── splits_metadata.json")
    
    print(f"\n🎯 Test completed! Check: {run_manager.run_dir}")

if __name__ == "__main__":
    test_data_splits_generation() 
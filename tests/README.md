# Tests

This directory contains test scripts for validating system functionality.

## Test Scripts

### `verify_gpu_setup.py`

**Purpose**: Verify GPU acceleration setup and device detection  
**Usage**: `python tests/verify_gpu_setup.py`  
**Use when**: After environment setup to confirm GPU acceleration is working

### `test_prediction_system.py`

**Purpose**: Test the standardized prediction file system  
**Usage**: `python tests/test_prediction_system.py`  
**Use when**: Validating prediction file generation and loading functionality

### `test_zoom_loading.py`

**Purpose**: Test zoom-aware data loading and data leakage prevention  
**Usage**: `python tests/test_zoom_loading.py`  
**Use when**: Validating data loading splits and ensuring no data leakage

## Running Tests

```bash
# Run all tests
cd tests
python verify_gpu_setup.py
python test_prediction_system.py
python test_zoom_loading.py

# Or run individual tests as needed
python tests/verify_gpu_setup.py  # From project root
```

These tests help verify that the system is properly configured and functioning correctly.

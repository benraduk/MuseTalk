# Test Scripts

This directory contains all test scripts for the MuseTalk project.

## Test Files

### `test_surgical_requirements.py`
Tests the surgical requirements validation system to ensure that the surgical integration eliminates conflicts successfully. This script validates core dependencies and checks for import compatibility.

### `test_enhanced_musetalk.py` 
Tests the enhanced MuseTalk preprocessing and inference pipeline with FaceFusion-style frame handling. Includes tests for enhanced preprocessing functions and inference capabilities.

### `test_ffmpeg.py`
Tests FFmpeg installation and functionality. Verifies that FFmpeg is properly installed and can be executed from the system path.

## Running Tests

To run the tests, execute them from the project root directory:

```bash
# Run individual tests
python tests/test_surgical_requirements.py
python tests/test_enhanced_musetalk.py
python tests/test_ffmpeg.py
```

Make sure you have activated the appropriate conda environment (e.g., 'braivtalk') before running the tests.
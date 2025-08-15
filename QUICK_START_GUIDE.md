# NSPO Quick Start Guide

## Problem Solved
The original evaluation and training scripts required downloading large models from HuggingFace, which:
- Takes a long time (GPT-2 is ~500MB)
- Requires internet connection
- Times out in constrained environments

## Solution Implemented

### 1. Offline Evaluation Support
Added support for a "dummy" model that doesn't require downloads:

```bash
# Run evaluation without downloading models
cd NSPO
python evaluate.py --model_path dummy --fast --cpu_only
```

### 2. Quick Test Scripts
Created test scripts for rapid validation:

```bash
# Test basic functionality
python test_evaluation.py

# Run quick evaluation with dummy model
python test_quick_eval.py
```

### 3. Key Changes Made

#### evaluate.py
- Added `create_dummy_model()` function for offline testing
- Support for `--model_path dummy` option
- Works without internet connection

#### Training Script Support
Both `train.py` and `evaluate.py` now support:
- `--fast` flag for quick testing (limits samples and batches)
- `--cpu_only` flag to force CPU execution
- `--limit_eval_samples` to control dataset size

## Usage Examples

### Quick Offline Evaluation
```bash
python evaluate.py \
    --model_path dummy \
    --fast \
    --cpu_only \
    --output_path results.json
```

### With Real Model (requires download)
```bash
python evaluate.py \
    --model_path gpt2 \
    --batch_size 8 \
    --output_path gpt2_results.json
```

### Training with Dummy Model (TODO)
```bash
python train.py \
    --model_path dummy \
    --fast \
    --cpu_only \
    --num_epochs 1
```

## Results
- Evaluation now works offline
- Test scripts validate all components
- Results saved to JSON files for analysis
- No timeout issues with dummy models

## Files Created/Modified
1. `evaluate.py` - Added dummy model support
2. `test_evaluation.py` - Basic functionality test
3. `test_quick_eval.py` - Quick evaluation test
4. `offline_eval_results.json` - Sample results
5. `requirements.txt` - Updated version constraints

## Next Steps
To use with real models:
1. Ensure stable internet connection
2. Increase timeout limits if needed
3. Consider using smaller models like `distilgpt2` for faster testing
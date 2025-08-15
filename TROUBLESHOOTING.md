# Troubleshooting Guide

## Common Issues and Solutions

### Installation Issues

#### 1. PyTorch Installation Fails
**Problem**: PyTorch installation fails or takes too long due to CUDA dependencies.

**Solution**:
```bash
# Use CPU-only version
pip install -r requirements-cpu.txt

# Or manually install CPU PyTorch
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

#### 2. Out of Memory Errors
**Problem**: Model training fails with CUDA out of memory or system memory errors.

**Solutions**:
- Use smaller batch size: `--batch_size 2`
- Use CPU-only mode: `--cpu_only`
- Use lightweight model: `--model_path gpt2` (smallest GPT-2)
- Limit sequence length: `--max_length 256`
- Use gradient accumulation: `--gradient_accumulation_steps 4`

#### 3. Slow Training/Evaluation
**Problem**: Training or evaluation takes too long.

**Solutions**:
```bash
# Use fast mode for testing
python train.py --fast --cpu_only

# Limit samples
python train.py --limit_train_samples 100 --limit_eval_samples 50

# Set maximum steps
python train.py --max_steps 50
```

### Model Loading Issues

#### 1. Model Download Fails
**Problem**: Cannot download models from Hugging Face.

**Solutions**:
- Check internet connection
- Set proxy if needed: `export HTTPS_PROXY=your_proxy`
- Use offline mode with local model path
- Use the default gpt2 model which is small and fast

#### 2. Tokenizer Errors
**Problem**: Tokenizer pad_token not set.

**Solution**: This is handled automatically in the code, but if issues persist:
```python
tokenizer.pad_token = tokenizer.eos_token
```

### Runtime Errors

#### 1. Import Errors
**Problem**: Modules cannot be imported.

**Solutions**:
```bash
# Verify installation
python verify_installation.py

# Reinstall dependencies
pip install --upgrade -r requirements.txt

# Check Python version (requires 3.10+)
python --version
```

#### 2. Path Not Found Errors
**Problem**: Directories or files not found.

**Solution**: The code now automatically creates required directories. If issues persist:
```python
python -c "from utils.paths import initialize_directories; initialize_directories()"
```

#### 3. CUDA Not Available
**Problem**: CUDA is not detected even with GPU.

**Solutions**:
- Use CPU mode: `--cpu_only`
- Check CUDA installation: `nvidia-smi`
- Verify PyTorch CUDA: `python -c "import torch; print(torch.cuda.is_available())"`

### Testing Issues

#### 1. Tests Fail
**Problem**: Pytest tests fail.

**Solutions**:
```bash
# Run only smoke tests
pytest tests/ -m smoke

# Run with verbose output
pytest tests/ -v -s

# Skip slow tests
pytest tests/ -m "not slow"
```

#### 2. Coverage Reports Not Generated
**Problem**: Coverage reports are missing.

**Solution**:
```bash
# Install coverage tools
pip install pytest-cov

# Run with coverage
pytest tests/ --cov=. --cov-report=html
```

### Docker Issues

#### 1. Docker Build Fails
**Problem**: Docker image build fails.

**Solutions**:
- Ensure Docker is installed and running
- Use BuildKit: `DOCKER_BUILDKIT=1 docker build -t nsrpo .`
- Clear Docker cache: `docker system prune -a`

#### 2. Container Runs Out of Memory
**Problem**: Docker container crashes due to memory limits.

**Solution**:
```bash
# Increase memory limit
docker run --memory="4g" --memory-swap="4g" nsrpo:latest
```

### Performance Issues

#### 1. Training is Too Slow
**Problem**: Training takes hours even for small datasets.

**Solutions**:
- Enable mixed precision: `--fp16` (GPU only)
- Use smaller model: `--model_path gpt2`
- Reduce logging frequency: `--log_every 500`
- Use synthetic data for testing

#### 2. Evaluation Hangs
**Problem**: Evaluation script hangs or takes forever.

**Solutions**:
```bash
# Limit evaluation batches
python evaluate.py --max_batches 10

# Use fast mode
python evaluate.py --fast --cpu_only

# Skip generation
python evaluate.py --no-generation
```

## Getting Help

If you encounter issues not covered here:

1. Check the [GitHub Issues](https://github.com/yourusername/NSPO/issues)
2. Run the verification script: `python verify_installation.py`
3. Check logs in `logs/` directory
4. Create a minimal reproducible example
5. Open a new issue with:
   - Python version
   - PyTorch version
   - Error message
   - Minimal code to reproduce

## Environment Variables

Set these environment variables if needed:

```bash
# Proxy settings
export HTTP_PROXY=your_proxy
export HTTPS_PROXY=your_proxy

# Hugging Face cache
export HF_HOME=/path/to/cache
export TRANSFORMERS_CACHE=/path/to/cache

# PyTorch settings
export TORCH_HOME=/path/to/torch/cache

# Disable telemetry
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
```

## Quick Diagnostic Commands

```bash
# Check system info
python -c "import platform; print(platform.platform())"

# Check Python version
python --version

# Check PyTorch
python -c "import torch; print(f'PyTorch: {torch.__version__}')"

# Check CUDA
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

# Check all dependencies
python verify_installation.py

# Run minimal test
python train.py --fast --cpu_only --max_steps 5
```
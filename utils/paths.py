"""
Path utility functions for NSRPO project.
Provides consistent path handling across the codebase.
"""

import os
from pathlib import Path
from datetime import datetime
from typing import Union, Optional

# Project root directory (where this utils folder is located)
PROJECT_ROOT = Path(__file__).parent.parent.absolute()

# Standard directories
DATA_DIR = PROJECT_ROOT / "data"
CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
LOGS_DIR = PROJECT_ROOT / "logs"
FIGURES_DIR = PROJECT_ROOT / "figures"
RESULTS_DIR = PROJECT_ROOT / "results"
CACHE_DIR = PROJECT_ROOT / ".cache"

def ensure_dir(path: Union[str, Path]) -> Path:
    """
    Ensure a directory exists, creating it if necessary.
    
    Args:
        path: Directory path to ensure exists
        
    Returns:
        Path object for the directory
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path

def get_project_root() -> Path:
    """Get the project root directory."""
    return PROJECT_ROOT

def get_data_dir() -> Path:
    """Get the data directory, creating it if necessary."""
    return ensure_dir(DATA_DIR)

def get_checkpoints_dir() -> Path:
    """Get the checkpoints directory, creating it if necessary."""
    return ensure_dir(CHECKPOINTS_DIR)

def get_outputs_dir() -> Path:
    """Get the outputs directory, creating it if necessary."""
    return ensure_dir(OUTPUTS_DIR)

def get_logs_dir() -> Path:
    """Get the logs directory, creating it if necessary."""
    return ensure_dir(LOGS_DIR)

def get_figures_dir() -> Path:
    """Get the figures directory, creating it if necessary."""
    return ensure_dir(FIGURES_DIR)

def get_results_dir() -> Path:
    """Get the results directory, creating it if necessary."""
    return ensure_dir(RESULTS_DIR)

def get_cache_dir() -> Path:
    """Get the cache directory, creating it if necessary."""
    return ensure_dir(CACHE_DIR)

def get_timestamped_filename(base_name: str, extension: str = None, 
                           directory: Union[str, Path] = None) -> Path:
    """
    Generate a timestamped filename to prevent overwriting.
    
    Args:
        base_name: Base name for the file
        extension: File extension (without dot)
        directory: Directory to place the file in
        
    Returns:
        Path object with timestamped filename
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if extension:
        if not extension.startswith('.'):
            extension = f'.{extension}'
        filename = f"{base_name}_{timestamp}{extension}"
    else:
        filename = f"{base_name}_{timestamp}"
    
    if directory:
        directory = ensure_dir(directory)
        return directory / filename
    else:
        return Path(filename)

def get_checkpoint_path(model_name: str, epoch: Optional[int] = None, 
                        timestamped: bool = True) -> Path:
    """
    Get a path for saving model checkpoints.
    
    Args:
        model_name: Name of the model
        epoch: Epoch number (optional)
        timestamped: Whether to add timestamp to filename
        
    Returns:
        Path for the checkpoint file
    """
    checkpoints_dir = get_checkpoints_dir()
    
    if epoch is not None:
        base_name = f"{model_name}_epoch_{epoch}"
    else:
        base_name = model_name
    
    if timestamped:
        return get_timestamped_filename(base_name, "pt", checkpoints_dir)
    else:
        return checkpoints_dir / f"{base_name}.pt"

def get_output_path(name: str, extension: str = "json", 
                   timestamped: bool = True) -> Path:
    """
    Get a path for saving output files.
    
    Args:
        name: Name for the output file
        extension: File extension
        timestamped: Whether to add timestamp to filename
        
    Returns:
        Path for the output file
    """
    outputs_dir = get_outputs_dir()
    
    if timestamped:
        return get_timestamped_filename(name, extension, outputs_dir)
    else:
        if not extension.startswith('.'):
            extension = f'.{extension}'
        return outputs_dir / f"{name}{extension}"

def get_log_path(name: str, timestamped: bool = True) -> Path:
    """
    Get a path for log files.
    
    Args:
        name: Name for the log file
        timestamped: Whether to add timestamp to filename
        
    Returns:
        Path for the log file
    """
    logs_dir = get_logs_dir()
    
    if timestamped:
        return get_timestamped_filename(name, "log", logs_dir)
    else:
        return logs_dir / f"{name}.log"

def get_figure_path(name: str, extension: str = "png", 
                   timestamped: bool = False) -> Path:
    """
    Get a path for saving figures.
    
    Args:
        name: Name for the figure
        extension: File extension (png, pdf, svg, etc.)
        timestamped: Whether to add timestamp to filename
        
    Returns:
        Path for the figure file
    """
    figures_dir = get_figures_dir()
    
    if timestamped:
        return get_timestamped_filename(name, extension, figures_dir)
    else:
        if not extension.startswith('.'):
            extension = f'.{extension}'
        return figures_dir / f"{name}{extension}"

def resolve_path(path: Union[str, Path], base: Optional[Path] = None) -> Path:
    """
    Resolve a path relative to a base directory or project root.
    
    Args:
        path: Path to resolve
        base: Base directory (defaults to project root)
        
    Returns:
        Resolved absolute path
    """
    path = Path(path)
    
    if path.is_absolute():
        return path
    
    if base is None:
        base = PROJECT_ROOT
    else:
        base = Path(base)
    
    return (base / path).resolve()

def find_file(filename: str, search_dirs: Optional[list] = None) -> Optional[Path]:
    """
    Find a file in common project directories.
    
    Args:
        filename: Name of the file to find
        search_dirs: List of directories to search (defaults to common dirs)
        
    Returns:
        Path to the file if found, None otherwise
    """
    if search_dirs is None:
        search_dirs = [
            PROJECT_ROOT,
            DATA_DIR,
            CHECKPOINTS_DIR,
            OUTPUTS_DIR,
            RESULTS_DIR,
        ]
    
    for directory in search_dirs:
        directory = Path(directory)
        if directory.exists():
            file_path = directory / filename
            if file_path.exists():
                return file_path
            
            # Also search subdirectories
            for subpath in directory.rglob(filename):
                return subpath
    
    return None

def clean_old_files(directory: Union[str, Path], pattern: str = "*", 
                   keep_recent: int = 5):
    """
    Clean old files from a directory, keeping only the most recent ones.
    
    Args:
        directory: Directory to clean
        pattern: File pattern to match
        keep_recent: Number of recent files to keep
    """
    directory = Path(directory)
    if not directory.exists():
        return
    
    files = sorted(directory.glob(pattern), key=lambda f: f.stat().st_mtime)
    
    if len(files) > keep_recent:
        for file in files[:-keep_recent]:
            try:
                file.unlink()
                print(f"Removed old file: {file}")
            except Exception as e:
                print(f"Error removing {file}: {e}")

# Initialize standard directories on import
def initialize_directories():
    """Initialize all standard project directories."""
    dirs = [
        DATA_DIR,
        CHECKPOINTS_DIR,
        OUTPUTS_DIR,
        LOGS_DIR,
        FIGURES_DIR,
        RESULTS_DIR,
        CACHE_DIR,
    ]
    
    for directory in dirs:
        ensure_dir(directory)
        
    # Create .gitignore in cache directory
    gitignore_path = CACHE_DIR / ".gitignore"
    if not gitignore_path.exists():
        gitignore_path.write_text("*\n!.gitignore\n")

# Initialize directories when module is imported
initialize_directories()
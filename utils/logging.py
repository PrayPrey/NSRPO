"""
Logging utilities for NSRPO project.
Provides standardized logging across the codebase.
"""

import logging
import sys
import time
from pathlib import Path
from typing import Optional, Union, Any
from functools import wraps
from datetime import datetime
import json


class ColoredFormatter(logging.Formatter):
    """Custom formatter with colors for console output."""
    
    COLORS = {
        'DEBUG': '\033[36m',    # Cyan
        'INFO': '\033[32m',     # Green
        'WARNING': '\033[33m',  # Yellow
        'ERROR': '\033[31m',    # Red
        'CRITICAL': '\033[35m', # Magenta
    }
    RESET = '\033[0m'
    
    def format(self, record):
        # Add color to level name
        levelname = record.levelname
        if levelname in self.COLORS:
            record.levelname = f"{self.COLORS[levelname]}{levelname}{self.RESET}"
        
        # Format the message
        result = super().format(record)
        
        # Reset levelname for other handlers
        record.levelname = levelname
        
        return result


class JSONFormatter(logging.Formatter):
    """JSON formatter for structured logging."""
    
    def format(self, record):
        log_obj = {
            'timestamp': datetime.utcnow().isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno
        }
        
        # Add extra fields if present
        if hasattr(record, 'extra_fields'):
            log_obj.update(record.extra_fields)
        
        return json.dumps(log_obj)


def setup_logger(
    name: str = None,
    level: Union[str, int] = "INFO",
    log_file: Optional[Union[str, Path]] = None,
    console: bool = True,
    format_style: str = "standard",  # "standard", "colored", "json"
    rotation: bool = False
) -> logging.Logger:
    """
    Set up a logger with specified configuration.
    
    Args:
        name: Logger name (None for root logger)
        level: Logging level
        log_file: Path to log file (optional)
        console: Whether to log to console
        format_style: Format style for logs
        rotation: Whether to use rotating file handler
        
    Returns:
        Configured logger
    """
    # Get or create logger
    logger = logging.getLogger(name)
    
    # Clear existing handlers
    logger.handlers = []
    
    # Set level
    if isinstance(level, str):
        level = getattr(logging, level.upper())
    logger.setLevel(level)
    
    # Create formatters
    if format_style == "json":
        formatter = JSONFormatter()
    elif format_style == "colored" and console:
        formatter = ColoredFormatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
    else:
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
    
    # Add console handler
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter if format_style != "colored" 
                                    else ColoredFormatter(
                                        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                                        datefmt='%Y-%m-%d %H:%M:%S'
                                    ))
        logger.addHandler(console_handler)
    
    # Add file handler
    if log_file:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        
        if rotation:
            from logging.handlers import RotatingFileHandler
            file_handler = RotatingFileHandler(
                log_file,
                maxBytes=10*1024*1024,  # 10MB
                backupCount=5
            )
        else:
            file_handler = logging.FileHandler(log_file)
        
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    # Prevent propagation to avoid duplicate logs
    logger.propagate = False
    
    return logger


def get_logger(name: str = None) -> logging.Logger:
    """
    Get a logger instance.
    
    Args:
        name: Logger name
        
    Returns:
        Logger instance
    """
    return logging.getLogger(name)


def log_execution_time(func=None, logger=None, level=logging.INFO):
    """
    Decorator to log function execution time.
    
    Args:
        func: Function to decorate
        logger: Logger to use (uses function's module logger if None)
        level: Logging level
        
    Returns:
        Decorated function
    """
    def decorator(f):
        @wraps(f)
        def wrapper(*args, **kwargs):
            # Get logger
            log = logger or logging.getLogger(f.__module__)
            
            # Start timer
            start_time = time.time()
            
            # Log start
            log.log(level, f"Starting {f.__name__}")
            
            try:
                # Execute function
                result = f(*args, **kwargs)
                
                # Calculate execution time
                execution_time = time.time() - start_time
                
                # Log completion
                log.log(level, f"Completed {f.__name__} in {execution_time:.2f} seconds")
                
                return result
            
            except Exception as e:
                # Calculate execution time
                execution_time = time.time() - start_time
                
                # Log error
                log.error(f"Error in {f.__name__} after {execution_time:.2f} seconds: {str(e)}")
                raise
        
        return wrapper
    
    if func is None:
        # Called with arguments
        return decorator
    else:
        # Called without arguments
        return decorator(func)


def log_memory_usage(func=None, logger=None, level=logging.INFO):
    """
    Decorator to log function memory usage.
    
    Args:
        func: Function to decorate
        logger: Logger to use
        level: Logging level
        
    Returns:
        Decorated function
    """
    def decorator(f):
        @wraps(f)
        def wrapper(*args, **kwargs):
            import psutil
            import os
            
            # Get logger
            log = logger or logging.getLogger(f.__module__)
            
            # Get initial memory
            process = psutil.Process(os.getpid())
            mem_before = process.memory_info().rss / 1024 / 1024  # MB
            
            try:
                # Execute function
                result = f(*args, **kwargs)
                
                # Get final memory
                mem_after = process.memory_info().rss / 1024 / 1024  # MB
                mem_diff = mem_after - mem_before
                
                # Log memory usage
                log.log(level, f"{f.__name__} memory usage: {mem_diff:+.2f} MB "
                            f"(before: {mem_before:.2f} MB, after: {mem_after:.2f} MB)")
                
                return result
            
            except Exception as e:
                # Get final memory
                mem_after = process.memory_info().rss / 1024 / 1024  # MB
                mem_diff = mem_after - mem_before
                
                # Log error with memory info
                log.error(f"Error in {f.__name__} with memory change {mem_diff:+.2f} MB: {str(e)}")
                raise
        
        return wrapper
    
    if func is None:
        return decorator
    else:
        return decorator(func)


class LogContext:
    """Context manager for temporary logging configuration."""
    
    def __init__(self, logger: logging.Logger, level: Optional[int] = None,
                 handler: Optional[logging.Handler] = None):
        """
        Initialize log context.
        
        Args:
            logger: Logger to modify
            level: Temporary logging level
            handler: Temporary handler to add
        """
        self.logger = logger
        self.level = level
        self.handler = handler
        self.original_level = None
        self.original_handlers = None
    
    def __enter__(self):
        """Enter context."""
        # Save original state
        self.original_level = self.logger.level
        self.original_handlers = self.logger.handlers.copy()
        
        # Apply temporary changes
        if self.level is not None:
            self.logger.setLevel(self.level)
        
        if self.handler is not None:
            self.logger.addHandler(self.handler)
        
        return self.logger
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context."""
        # Restore original state
        self.logger.setLevel(self.original_level)
        
        if self.handler is not None:
            self.logger.removeHandler(self.handler)


class ProgressLogger:
    """Logger for tracking progress of long-running operations."""
    
    def __init__(self, total: int, logger: logging.Logger = None,
                 desc: str = "Processing", level: int = logging.INFO):
        """
        Initialize progress logger.
        
        Args:
            total: Total number of items
            logger: Logger to use
            desc: Description of the operation
            level: Logging level
        """
        self.total = total
        self.current = 0
        self.logger = logger or logging.getLogger(__name__)
        self.desc = desc
        self.level = level
        self.start_time = time.time()
        
    def update(self, n: int = 1):
        """Update progress."""
        self.current += n
        percentage = (self.current / self.total) * 100
        elapsed = time.time() - self.start_time
        
        if self.current < self.total:
            eta = (elapsed / self.current) * (self.total - self.current)
            self.logger.log(self.level, 
                          f"{self.desc}: {self.current}/{self.total} "
                          f"({percentage:.1f}%) - ETA: {eta:.1f}s")
        else:
            self.logger.log(self.level,
                          f"{self.desc}: Completed {self.total} items "
                          f"in {elapsed:.1f}s")
    
    def __enter__(self):
        """Enter context."""
        self.logger.log(self.level, f"{self.desc}: Starting {self.total} items")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context."""
        if exc_type is None:
            elapsed = time.time() - self.start_time
            self.logger.log(self.level, 
                          f"{self.desc}: Completed in {elapsed:.1f}s")


# Default logger configuration
def configure_default_logging(level: str = "INFO", log_dir: Optional[Path] = None):
    """
    Configure default logging for the entire project.
    
    Args:
        level: Default logging level
        log_dir: Directory for log files
    """
    # Set up root logger
    root_logger = setup_logger(
        level=level,
        console=True,
        format_style="colored"
    )
    
    # Set up file logging if directory provided
    if log_dir:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # Add file handler for all logs
        file_handler = logging.FileHandler(log_dir / "nsrpo.log")
        file_handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        ))
        root_logger.addHandler(file_handler)
        
        # Add separate error log
        error_handler = logging.FileHandler(log_dir / "errors.log")
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(pathname)s:%(lineno)d - %(message)s'
        ))
        root_logger.addHandler(error_handler)


if __name__ == "__main__":
    # Test logging utilities
    print("Testing logging utilities...")
    
    # Configure default logging
    configure_default_logging(level="DEBUG")
    
    # Get logger
    logger = get_logger("test")
    
    # Test different log levels
    logger.debug("Debug message")
    logger.info("Info message")
    logger.warning("Warning message")
    logger.error("Error message")
    
    # Test execution time decorator
    @log_execution_time
    def slow_function():
        time.sleep(0.5)
        return "Done"
    
    result = slow_function()
    
    # Test progress logger
    with ProgressLogger(10, logger, "Test Progress") as progress:
        for i in range(10):
            time.sleep(0.1)
            progress.update()
    
    print("Logging utilities test completed!")
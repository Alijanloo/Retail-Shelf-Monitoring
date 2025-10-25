import logging
import logging.config
import sys
from pathlib import Path
from typing import Optional

import yaml

from retail_shelf_monitoring import DEFAULT_PATH


def load_logging_config():
    config_path = DEFAULT_PATH / "config.yaml"
    try:
        with open(config_path, "r") as file:
            config = yaml.safe_load(file)
            return config.get("logging", {})
    except FileNotFoundError:
        return {}
    except Exception:
        return {}


class ColoredFormatter(logging.Formatter):
    COLORS = {
        "DEBUG": "\033[36m",  # Cyan
        "INFO": "\033[32m",  # Green
        "WARNING": "\033[33m",  # Yellow
        "ERROR": "\033[31m",  # Red
        "CRITICAL": "\033[35m",  # Magenta
    }

    BOLD = "\033[1m"
    RESET = "\033[0m"

    def __init__(self, *args, use_colors=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_colors = use_colors and self._supports_color()

    def _supports_color(self):
        """Check if the terminal supports color output."""
        return (
            hasattr(sys.stderr, "isatty")
            and sys.stderr.isatty()
            and hasattr(sys.stdout, "isatty")
            and sys.stdout.isatty()
        )

    def format(self, record):
        """Format the log record with colors."""
        if not self.use_colors:
            return super().format(record)

        original_format = super().format(record)
        level_color = self.COLORS.get(record.levelname, "")

        if level_color:
            parts = original_format.split(" - ", 5)

            if len(parts) >= 6:
                header_parts = parts[:5]
                message = parts[5]
                colored_header = (
                    f"{self.BOLD}{level_color}{' - '.join(header_parts)}{self.RESET}"
                )
                return f"{colored_header} - {message}"
            else:
                colored_level = (
                    f"{self.BOLD}{level_color}{record.levelname}{self.RESET}"
                )
                return original_format.replace(record.levelname, colored_level, 1)

        return original_format


class LoggerFactory:
    _initialized = False
    _log_dir: Optional[Path] = None

    @classmethod
    def setup_logging(
        cls,
        log_level: str = "INFO",
        log_dir: Optional[str] = None,
        console_output: bool = True,
        file_output: bool = True,
        format_type: str = "detailed",
        use_colors: bool = True,
    ) -> None:
        """
        Configure logging for the entire application.

        Args:
            log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
            log_dir: Directory for log files. If None, uses project root/logs
            console_output: Whether to log to console
            file_output: Whether to log to files
            format_type: Format type ('detailed', 'simple', 'json')
            use_colors: Whether to use colors in console output
        """
        if cls._initialized:
            return

        if log_dir is None:
            project_root = Path(__file__).parents[2]
            cls._log_dir = project_root / "logs"
        else:
            cls._log_dir = Path(log_dir)

        cls._log_dir.mkdir(exist_ok=True)

        formats = {
            "simple": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            "detailed": (
                "%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - "
                "%(funcName)s() - %(message)s"
            ),
            "json": (
                "%(asctime)s | %(name)s | %(levelname)s | %(filename)s:%(lineno)d | "
                "%(funcName)s | %(message)s"
            ),
        }

        log_format = formats.get(format_type, formats["detailed"])

        config = {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "standard": {"format": log_format, "datefmt": "%Y-%m-%d %H:%M:%S"},
                "colored": {
                    "()": ColoredFormatter,
                    "format": log_format,
                    "datefmt": "%Y-%m-%d %H:%M:%S",
                    "use_colors": use_colors,
                },
            },
            "handlers": {},
            "loggers": {
                "retail_shelf_monitoring": {
                    "level": log_level,
                    "handlers": [],
                    "propagate": False,
                },
                "root": {"level": "WARNING", "handlers": []},
            },
        }

        if console_output:
            config["handlers"]["console"] = {
                "class": "logging.StreamHandler",
                "level": log_level,
                "formatter": "colored",
                "stream": "ext://sys.stdout",
            }
            config["loggers"]["retail_shelf_monitoring"]["handlers"].append("console")
            config["loggers"]["root"]["handlers"].append("console")

        if file_output:
            config["handlers"]["file"] = {
                "class": "logging.handlers.RotatingFileHandler",
                "level": log_level,
                "formatter": "standard",
                "filename": str(cls._log_dir / "retail_shelf_monitoring.log"),
                "maxBytes": 10485760,
                "backupCount": 5,
            }

            config["handlers"]["error_file"] = {
                "class": "logging.handlers.RotatingFileHandler",
                "level": "ERROR",
                "formatter": "standard",
                "filename": str(cls._log_dir / "retail_shelf_monitoring_errors.log"),
                "maxBytes": 10485760,
                "backupCount": 5,
            }

            config["loggers"]["retail_shelf_monitoring"]["handlers"].extend(
                ["file", "error_file"]
            )
            config["loggers"]["root"]["handlers"].extend(["file", "error_file"])

        logging.config.dictConfig(config)
        cls._initialized = True

        logger = logging.getLogger("retail_shelf_monitoring.logging_config")
        logger.info(
            f"Logging initialized - Level: {log_level}, Log dir: {cls._log_dir}"
        )

    @classmethod
    def get_logger(cls, name: str) -> logging.Logger:
        """
        Get a logger instance for the given name.

        Args:
            name: Logger name, typically __name__ from the calling module

        Returns:
            Configured logger instance
        """
        if not cls._initialized:
            logging_config = load_logging_config()
            cls.setup_logging(
                log_level=logging_config.get("level", "INFO"),
                console_output=logging_config.get("console_output", True),
                file_output=logging_config.get("file_output", True),
                format_type=logging_config.get("format_type", "detailed"),
                use_colors=logging_config.get("use_colors", True),
            )

        if not name.startswith("retail_shelf_monitoring"):
            if name == "__main__":
                name = "retail_shelf_monitoring.main"
            else:
                name = f"retail_shelf_monitoring.{name}"

        return logging.getLogger(name)

    @classmethod
    def get_log_directory(cls) -> Optional[Path]:
        return cls._log_dir


def get_logger(name: str = None) -> logging.Logger:
    """
    Get a logger instance. Convenience function for easy importing.

    Args:
        name: Logger name. If None, uses the calling module's __name__

    Returns:
        Configured logger instance
    """
    if name is None:
        frame = sys._getframe(1)
        name = frame.f_globals.get("__name__", "unknown")

    return LoggerFactory.get_logger(name)


def setup_logging(**kwargs) -> None:
    LoggerFactory.setup_logging(**kwargs)


class LogLevel:
    """Context manager for temporarily changing log level."""

    def __init__(self, logger: logging.Logger, level: str):
        self.logger = logger
        self.new_level = getattr(logging, level.upper())
        self.old_level = None

    def __enter__(self):
        self.old_level = self.logger.level
        self.logger.setLevel(self.new_level)
        return self.logger

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.logger.setLevel(self.old_level)


def log_function_call(logger: logging.Logger = None, level: str = "DEBUG"):
    """
    Decorator to log function entry and exit.

    Args:
        logger: Logger to use. If None, creates one based on function module
        level: Log level to use
    """

    def decorator(func):
        nonlocal logger
        if logger is None:
            logger = get_logger(func.__module__)

        log_level = getattr(logging, level.upper())

        def wrapper(*args, **kwargs):
            logger.log(log_level, f"Entering {func.__name__}")
            try:
                result = func(*args, **kwargs)
                logger.log(log_level, f"Exiting {func.__name__}")
                return result
            except Exception as e:
                logger.error(f"Exception in {func.__name__}: {e}")
                raise

        return wrapper

    return decorator


def log_execution_time(logger: logging.Logger = None, level: str = "INFO"):
    """
    Decorator to log function execution time.

    Args:
        logger: Logger to use. If None, creates one based on function module
        level: Log level to use
    """

    def decorator(func):
        nonlocal logger
        if logger is None:
            logger = get_logger(func.__module__)

        log_level = getattr(logging, level.upper())

        def wrapper(*args, **kwargs):
            import time

            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                execution_time = time.time() - start_time
                logger.log(
                    log_level, f"{func.__name__} executed in {execution_time:.3f}s"
                )
                return result
            except Exception as e:
                execution_time = time.time() - start_time
                logger.error(f"{func.__name__} failed after {execution_time:.3f}s: {e}")
                raise

        return wrapper

    return decorator

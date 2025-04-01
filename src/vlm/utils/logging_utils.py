import logging
import os
from logging import LogRecord
from typing import override


class RankZeroFilter(logging.Filter):
    @override
    def filter(self, record: logging.LogRecord) -> bool:
        return os.environ.get("LOCAL_RANK", "0") == "0"


class ColoredFormatter(logging.Formatter):
    @override
    def format(self, record: logging.LogRecord) -> str:
        # Create a copy of the record to avoid modifying the original
        # This is important because the same record might be used by multiple formatters
        colored_record: LogRecord = logging.makeLogRecord(record.__dict__)

        # Get the original message
        message: str = colored_record.getMessage()

        # Add color based on level
        if colored_record.levelno == logging.INFO:
            colored_message: str = f"[bold green]{message}[/bold green]"
        elif colored_record.levelno == logging.WARNING:
            colored_message = f"[bold yellow]{message}[/bold yellow]"
        elif colored_record.levelno == logging.ERROR:
            colored_message = f"[bold red]{message}[/bold red]"
        elif colored_record.levelno == logging.DEBUG:
            colored_message = f"[bold blue]{message}[/bold blue]"
        else:
            colored_message = message

        # Replace the message in the copied record
        colored_record.msg = colored_message

        # Use the standard formatter to format the whole record
        return super().format(colored_record)

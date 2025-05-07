import logging
from typing import Any, override

import transformers
from rich.console import Console
from rich.logging import RichHandler
from rich.progress import (
    BarColumn,
    Progress,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.table import Table
from torch.utils.data import DataLoader

console = Console()
rich_handler = RichHandler(console=console)
logger: logging.Logger = logging.getLogger(name=__name__)


def has_length(dataloader: DataLoader) -> bool:
    try:
        len(dataloader)
        return True
    except (TypeError, ValueError, AttributeError):
        return False


class ProgressCallback(transformers.TrainerCallback):
    def __init__(self, max_str_len: int = 100):
        """
        Initialize the callback with optional max_str_len parameter to control string truncation length.

        Args:
            max_str_len (`int`):
                Maximum length of strings to display in logs.
                Longer strings will be truncated with a message.
        """
        self.max_str_len: int = max_str_len
        self.console: Console = Console()
        self.current_step: int = 0
        self.training_progress: Progress | None = None
        self.train_task_id: Any = None
        self.prediction_progress: Progress | None = None
        self.eval_task_id: Any = None

    @override
    def on_train_begin(self, args, state, control, **kwargs):  # pyright: ignore
        if state.is_world_process_zero:
            self.training_progress = Progress(
                TextColumn("[bold blue]{task.description}"),
                BarColumn(bar_width=None),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                TimeElapsedColumn(),
                TimeRemainingColumn(),
                console=self.console,
                expand=True,
            )
            self.train_task_id = self.training_progress.add_task(
                "[green]Training", total=state.max_steps
            )
            self.training_progress.start()
        self.current_step = 0

    @override
    def on_step_end(self, args, state, control, **kwargs):  # pyright: ignore
        if state.is_world_process_zero and self.training_progress is not None:
            steps_completed = state.global_step - self.current_step
            self.training_progress.update(self.train_task_id, advance=steps_completed)
            self.current_step = state.global_step

    @override
    def on_prediction_step(self, args, state, control, eval_dataloader=None, **kwargs):  # pyright: ignore
        if state.is_world_process_zero and has_length(eval_dataloader):
            if self.prediction_progress is None:
                self.prediction_progress = Progress(
                    TextColumn("[bold yellow]{task.description}"),
                    BarColumn(bar_width=None),
                    TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                    console=self.console,
                    expand=True,
                )
                self.eval_task_id = self.prediction_progress.add_task(
                    "[yellow]Evaluating", total=len(eval_dataloader)
                )
                self.prediction_progress.start()
            else:
                self.prediction_progress.update(self.eval_task_id, advance=1)

    @override
    def on_evaluate(self, args, state, control, **kwargs):  # pyright: ignore
        if state.is_world_process_zero and self.prediction_progress is not None:
            self.prediction_progress.stop()
            self.prediction_progress = None

    @override
    def on_predict(self, args, state, control, **kwargs):  # pyright: ignore
        if state.is_world_process_zero and self.prediction_progress is not None:
            self.prediction_progress.stop()
            self.prediction_progress = None

    @override
    def on_log(self, args, state, control, logs=None, **kwargs):  # pyright: ignore
        if state.is_world_process_zero:
            shallow_logs = {}
            for k, v in logs.items():  # pyright: ignore
                if isinstance(v, str) and len(v) > self.max_str_len:
                    shallow_logs[k] = (
                        f"[String too long to display, length: {len(v)} > {self.max_str_len}. "
                        "Consider increasing `max_str_len` if needed.]"
                    )
                elif isinstance(v, float):
                    if abs(v) < 0.0001 and v != 0.0:
                        shallow_logs[k] = f"{v:.8f}"
                    else:
                        shallow_logs[k] = f"{v:.4f}"
                else:
                    shallow_logs[k] = v
            _ = shallow_logs.pop("total_flos", None)

            if "epoch" in shallow_logs and isinstance(shallow_logs["epoch"], float):
                shallow_logs["epoch"] = f"{shallow_logs['epoch']:.4f}"

            table = Table(show_header=True, header_style="bold magenta")
            table.add_column("Metric")
            table.add_column("Value")

            for key, value in shallow_logs.items():
                table.add_row(key, str(value))

            if self.training_progress is not None:
                self.training_progress.stop()
                self.console.print(table)
                self.training_progress.start()
            else:
                self.console.print(table)

    @override
    def on_train_end(self, args, state, control, **kwargs):  # pyright: ignore
        if state.is_world_process_zero:
            if self.training_progress is not None:
                self.training_progress.stop()
                self.training_progress = None
            self.console.print("[bold green]Training completed![/bold green]")

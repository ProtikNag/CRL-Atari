"""
Logging utilities: TensorBoard, CSV, and console logging.
"""

import os
import csv
import logging
from typing import Dict, Any, Optional
from datetime import datetime

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    SummaryWriter = None


class Logger:
    """Multi-backend logger supporting console, CSV, and TensorBoard.

    Args:
        log_dir: Directory for log files.
        experiment_name: Name of the experiment.
        use_tensorboard: Whether to enable TensorBoard logging.
        use_wandb: Whether to enable Weights & Biases logging.
        wandb_project: W&B project name.
        wandb_entity: W&B entity/team name.
    """

    def __init__(
        self,
        log_dir: str,
        experiment_name: str = "experiment",
        use_tensorboard: bool = True,
        use_wandb: bool = False,
        wandb_project: str = "crl-atari",
        wandb_entity: Optional[str] = None,
    ):
        self.log_dir = os.path.join(log_dir, experiment_name)
        os.makedirs(self.log_dir, exist_ok=True)

        # Console logger
        self.console_logger = logging.getLogger(experiment_name)
        self.console_logger.setLevel(logging.INFO)
        if not self.console_logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                "[%(asctime)s] %(levelname)s - %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
            handler.setFormatter(formatter)
            self.console_logger.addHandler(handler)

            # Also log to file
            fh = logging.FileHandler(os.path.join(self.log_dir, "training.log"))
            fh.setFormatter(formatter)
            self.console_logger.addHandler(fh)

        # TensorBoard
        self.tb_writer = None
        if use_tensorboard and SummaryWriter is not None:
            self.tb_writer = SummaryWriter(log_dir=self.log_dir)

        # CSV log
        self.csv_path = os.path.join(self.log_dir, "metrics.csv")
        self._csv_initialized = False

        # W&B
        self.wandb_run = None
        if use_wandb:
            try:
                import wandb

                self.wandb_run = wandb.init(
                    project=wandb_project,
                    entity=wandb_entity,
                    name=experiment_name,
                    dir=self.log_dir,
                )
            except ImportError:
                self.console_logger.warning("wandb not installed, skipping.")

    def log_scalar(self, tag: str, value: float, step: int) -> None:
        """Log a scalar metric."""
        if self.tb_writer:
            self.tb_writer.add_scalar(tag, value, step)
        if self.wandb_run:
            import wandb

            wandb.log({tag: value}, step=step)

    def log_scalars(self, metrics: Dict[str, float], step: int) -> None:
        """Log multiple scalar metrics at once."""
        for tag, value in metrics.items():
            self.log_scalar(tag, value, step)
        # Also write to CSV
        self._write_csv(metrics, step)

    def _write_csv(self, metrics: Dict[str, float], step: int) -> None:
        """Append metrics to CSV log file."""
        row = {"step": step, **metrics}
        write_header = not self._csv_initialized
        with open(self.csv_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            if write_header:
                writer.writeheader()
                self._csv_initialized = True
            writer.writerow(row)

    def info(self, msg: str) -> None:
        """Log info message to console and file."""
        self.console_logger.info(msg)

    def warning(self, msg: str) -> None:
        """Log warning message."""
        self.console_logger.warning(msg)

    def close(self) -> None:
        """Close all logging backends."""
        if self.tb_writer:
            self.tb_writer.close()
        if self.wandb_run:
            import wandb

            wandb.finish()


def setup_logger(
    log_dir: str = "results/logs",
    experiment_name: Optional[str] = None,
    use_tensorboard: bool = True,
    use_wandb: bool = False,
    wandb_project: str = "crl-atari",
    wandb_entity: Optional[str] = None,
) -> Logger:
    """Create and return a Logger instance.

    Args:
        log_dir: Directory for logs.
        experiment_name: Name of experiment. Auto-generated if None.
        use_tensorboard: Enable TensorBoard.
        use_wandb: Enable W&B.
        wandb_project: W&B project name.
        wandb_entity: W&B entity.

    Returns:
        Configured Logger instance.
    """
    if experiment_name is None:
        experiment_name = datetime.now().strftime("run_%Y%m%d_%H%M%S")

    return Logger(
        log_dir=log_dir,
        experiment_name=experiment_name,
        use_tensorboard=use_tensorboard,
        use_wandb=use_wandb,
        wandb_project=wandb_project,
        wandb_entity=wandb_entity,
    )

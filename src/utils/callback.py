from pathlib import Path
from typing import Any, Optional, Sequence

import pytorch_lightning as pl
import torch
from pytorch_lightning import Callback
from pytorch_lightning.callbacks import BasePredictionWriter
from pytorch_lightning.utilities.types import STEP_OUTPUT


class PredictionWriter(BasePredictionWriter):
    def write_on_epoch_end(self, *args, **kwargs) -> None:
        raise NotImplementedError

    def __init__(self, output_dir):
        super().__init__(write_interval="batch")
        self.output_dir = Path(output_dir) / "predictions"
        self.output_dir.mkdir(exist_ok=True, parents=True)

    def write_on_batch_end(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        prediction: Any,
        batch_indices: Optional[Sequence[int]],
        batch: Any,
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        torch.save(prediction, self.output_dir / f"b{batch_idx}_r{trainer.global_rank}.pt")


class TrainValidationWriter(Callback):
    def __init__(self, output_dir):
        super().__init__()
        self.output_dir = Path(output_dir) / "predictions"
        self.output_dir.mkdir(exist_ok=True, parents=True)

    def write_on_disk(
        self, trainer: "pl.Trainer", target: Any, batch_idx: int = None, dataloader_idx: int = None
    ):
        output_path = self.output_dir
        filename = f"b{batch_idx}" if batch_idx is not None else ""
        filename += f"_d{dataloader_idx}" if dataloader_idx is not None else ""
        filename += f"_r{trainer.local_rank}" if trainer.local_rank is not None else ""
        filename += ".pt"
        torch.save(target, output_path / filename)

    def on_train_batch_end(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        outputs: STEP_OUTPUT,
        batch: Any,
        batch_idx: int,
    ) -> None:
        self.write_on_disk(trainer=trainer, batch_idx=batch_idx, target=outputs["save"])

    def on_validation_batch_end(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        outputs: Optional[STEP_OUTPUT],
        batch: Any,
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        self.write_on_disk(
            trainer=trainer,
            batch_idx=batch_idx,
            dataloader_idx=dataloader_idx,
            target=outputs["save"],
        )

    def on_test_batch_end(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        outputs: Optional[STEP_OUTPUT],
        batch: Any,
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        self.on_validation_batch_end(trainer, pl_module, outputs, batch, batch_idx, dataloader_idx)

    def on_predict_batch_end(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        outputs: Any,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        self.on_validation_batch_end(trainer, pl_module, outputs, batch, batch_idx, dataloader_idx)

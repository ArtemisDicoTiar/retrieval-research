import copy
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd
import pytorch_lightning as pl
import torch
import wandb
from pytorch_lightning.utilities.types import (
    EVAL_DATALOADERS,
    STEP_OUTPUT,
    TRAIN_DATALOADERS,
)
from torch import Tensor, nn
from torch.optim import Optimizer, lr_scheduler
from torchmetrics import MeanMetric
from transformers import DistilBertTokenizer, optimization

from src.metrics.eval import EvaluationMetric
from src.models.components import RetrieverBase


class RetrievalModel(pl.LightningModule):
    """if LightningDataModule is injected to trainer, following methods just need pass.

    if not, following methods must be implemented. However, this project uses LightningDataModule.
    So, following dataloader methods became just boilerplate of LightningModule.
    """

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        pass

    def test_dataloader(self) -> EVAL_DATALOADERS:
        pass

    def val_dataloader(self) -> EVAL_DATALOADERS:
        pass

    def predict_dataloader(self) -> EVAL_DATALOADERS:
        pass

    def __init__(
        self,
        model: RetrieverBase,
        train_loss: nn.Module,
        val_loss: EvaluationMetric,
        optimizer: Optimizer,
        scheduler: lr_scheduler,
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        # teacher is not needed for inference and evaluation.
        self.save_hyperparameters(logger=False, ignore=["train_loss", "val_loss"])

        self.model = model

        self.train_criterion = train_loss
        self.test_criterion: EvaluationMetric = val_loss
        self.val_criterion: EvaluationMetric = copy.deepcopy(val_loss)

        # for averaging loss across batches
        self.train_loss = MeanMetric()
        self.test_loss = MeanMetric()
        self.val_loss = MeanMetric()

    # hyper parameters
    def configure_optimizers(self):
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Examples:
            https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#configure-optimizers
        """
        optimizer = self.hparams.optimizer(params=self.parameters())
        scheduler = self.hparams.scheduler(optimizer=optimizer)

        return {
            "optimizer": optimizer,
            "interval": "step",
            "frequency": 1,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",  # or 'epoch'
                # "monitor": "loss",
                "frequency": 1,
            },
        }

    # For validation, test, prediction modes some model's need to change its internal setting (eg. peq-colbert)
    # this method calls `{student model}.set_validation()`
    def on_validation_start(self) -> None:
        self.model.set_validation()

    def on_test_start(self) -> None:
        self.on_validation_start()

    def on_predict_start(self) -> None:
        self.on_validation_start()

    # @@@@@@@@@@@@@@@@@ TRAINING methods @@@@@@@@@@@@@@@@@ #
    def training_step(self, batch: Tuple[Tensor, Tensor, Tensor], batch_idx: int):
        """This method defines how model is trained without DISTILLATION.

        :param batch: (features, labels, idx)
            features: query_feature, (pos_pas_feature, neg_pas_feature)
            labels: answer of dataset. (margin between q-pos_doc & q-neg_doc)
            idx: this is order of dataset that each model gets. (this can be same as batch_idx but mostly different)
        :param batch_idx: index from dataset batch
        :return:
        """
        queries, docs, labels = batch

        query_emb, pas_emb = self.model.encode(queries, docs)

        score_result = self.model.score(query_emb, pas_emb)

        losses: Dict[str, Tensor] = self.train_criterion(score_result, labels)
        return losses

    def training_step_end(self, output: dict) -> STEP_OUTPUT:
        """This method collects output if ddp is enabled.

        :param output: this output is from training_step
        :return:
        """
        gathered_output = dict(map(lambda i: (i[0], i[1].mean()), output.items()))
        self.log_dict(gathered_output)
        return gathered_output

    # @@@@@@@@@@@@@@@@@ VALIDATION methods @@@@@@@@@@@@@@@@@ #
    def validation_step(self, batch: Any, batch_idx: int, dataloader_idx: int) -> Dict[str, Any]:
        """Validation step.

        :param batch: (features, labels, idx)
            features: (query, passage)
                # this is different from training.
                # training mode uses positive passage and negative passage.
            labels: answer (score of query-passage, from BEIR)
            idx: same from training_step method
        :param batch_idx: same from training_step method
        :return: dictionary output passed to `self.validation_step_end`
        """
        queries, docs, labels = batch

        query_emb, pas_emb = self.model.encode(queries, docs)
        score_res = self.model.score(query_emb, pas_emb)
        score_res.update(
            {
                "batch_idx": batch_idx,
                "dataloader_idx": dataloader_idx,
                "labels": labels,
                "scores": score_res,
            }
        )

        return score_res

    def validation_step_end(self, outputs: dict) -> None:
        """This method updates evaluation metrics.

        :param outputs: this output is the result from `self.validation_step`
        """
        self.val_criterion.update(
            outputs["batch_idx"],
            outputs["dataloader_idx"],
            outputs["relevance"],
            outputs["labels"],
            outputs["scores"],
        )

    def on_validation_epoch_end(self) -> None:
        """This method is used for calling pytorch metric instance to calculate final value.

        After calculation, it logs the evaluation metric.
        """
        metric = self.val_criterion.compute()
        self.log_dict(metric)
        self.val_criterion.reset()

    # @@@@@@@@@@@@@@@@@ TEST methods @@@@@@@@@@@@@@@@@ #
    # following methods exactly do same logic as validation
    def test_step(self, batch: Any, batch_idx: int):
        return self.validation_step(batch, batch_idx)

    def test_step_end(self, outputs: dict) -> None:
        self.test_criterion.update(
            outputs["idxs"],
            outputs["relevance"],
            outputs["labels"],
        )

    def on_test_epoch_end(self) -> None:
        metric = self.test_criterion.compute()
        self.log_dict(metric)
        self.test_criterion.reset()

    # @@@@@@@@@@@@@@@@@ PREDICTION methods @@@@@@@@@@@@@@@@@ #
    def predict_step(
        self, batch: Any, batch_idx: int, **kwargs
    ) -> Union[Dict[str, Dict[str, Tensor]], pd.DataFrame]:
        """This method is used for inference.

        :param batch: (features, labels, idx)
            features: (query, passage)
                # this is different from training.
                # training mode uses positive passage and negative passage.
            labels: answer (score of query-passage, from BEIR)
            idx: same from training_step method
        :param batch_idx: same from training_step method
        :param kwargs: ???
        :return: {
            "query": {
                "input_ids": ...,
                "encoded_embeddings": ...
            },
            "passage": {
                "input_ids": ...,
                "encoded_embeddings": ...
            },
            "score": {
                "relevance": ...,
                "label": ...
            }
        }
        """
        features, labels, _, _ = batch

        query_emb, pas_emb = self.student.encode(query=features[0], passage1=features[1])
        score_res = self.student.score(query_emb, pas_emb)
        score_res.update({"labels": labels})

        output = {
            "query": {
                "input_ids": features[0]["input_ids"],
                "encoded_embeddings": query_emb["encoded_embeddings"].detach(),
            },
            "passage": {
                "input_ids": features[1]["input_ids"],
                "encoded_embeddings": pas_emb["encoded_embeddings"].detach(),
            },
            "score": {
                "relevance": score_res["relevance"].detach(),
                "label": score_res["labels"].detach(),
            },
        }
        return output

    def on_predict_epoch_end(self, results: List[List[Dict[str, Dict[str, Tensor]]]]) -> None:
        """This method is used for decoding some tensors in form of human-readable.

        :param results: list of list of `self.predict_step` output
        """
        tokenizer: DistilBertTokenizer = self.trainer.datamodule.tokenizer.tokenizer

        # just taking first random result
        result = results[0][0:1]

        # concatenating list of results
        output = dict(
            map(
                lambda category: (
                    category,
                    dict(
                        map(
                            lambda key: (
                                key,
                                torch.cat(list(map(lambda batch: batch[category][key], result))),
                            ),
                            result[0][category].keys(),
                        )
                    ),
                ),
                result[0].keys(),
            )
        )
        # unnecessary token remove & token_ids -> text
        remove_sep_mask = (
            lambda txt: txt.replace(" [SEP]", "").replace(" [MASK]", "").replace(" [PAD]", "")
        )
        # decoding query and passage
        query_text = list(
            map(remove_sep_mask, tokenizer.batch_decode(output["query"]["input_ids"]))
        )
        passage_text = list(
            map(remove_sep_mask, tokenizer.batch_decode(output["passage"]["input_ids"]))
        )

        # creating wandb table for visualisation
        table = wandb.Table(
            columns=[str(i) for i in range(output["query"]["encoded_embeddings"].shape[-1])],
            data=output["query"]["encoded_embeddings"].cpu().tolist()
            + output["passage"]["encoded_embeddings"].cpu().tolist(),
        )
        table.add_column(name="dtype", data=["q"] * len(query_text) + ["p"] * len(passage_text))
        table.add_column(name="text", data=query_text + passage_text)
        table.add_column(name="relevance", data=output["score"]["relevance"].cpu().tolist() * 2)
        table.add_column(name="label", data=output["score"]["label"].cpu().tolist() * 2)

        wandb.log({"output": table})

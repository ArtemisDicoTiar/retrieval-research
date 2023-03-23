from collections import Counter
from typing import Dict, List, Optional, Union

import pytorch_lightning as pl
from pytorch_lightning.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
from torch import Tensor
from torch.utils.data import DataLoader

from src.data2tensor.tokenizer import BaseTokenizer
from src.datamodule.beir.datamodule import BEIRDataset
from src.datamodule.beir.downloader import BEIRDownloader
from src.datamodule.modules.dataloader import GenericDataLoader
from src.datamodule.utils import InputData, get_terms, ssh_connector


class RetrievalDataModule(pl.LightningDataModule):
    def __init__(
        self,
        shuffle_train: bool,
        train_dataset: str = "msmarco",
        test_datasets: List[str] = None,
        beir_data_dir: str = None,
        tokenizer: BaseTokenizer = None,
        train_max_step: int = 140_000,
        train_batch_size: int = 32,
        test_batch_size: int = 64,
        workers: int = 64,
    ):
        super().__init__()

        # self.query_term_counts = Counter()

        self.train_dataset = train_dataset
        self.test_datasets = test_datasets

        self.beir_data_dir = beir_data_dir

        self.shuffle_train = shuffle_train
        self.train_max_step = train_max_step
        self.test_data = None
        self.train_data = None
        self.val_data = None
        self.tokenizer = tokenizer

        self.train_batch_size = train_batch_size
        self.test_batch_size = test_batch_size

        self.workers = workers

    def prepare_data(self) -> None:
        """This method prepares selected dataset from GPL and BEIR dataset url on desired path."""
        # TRAIN-DATASET
        BEIRDownloader(data_dir=self.beir_data_dir, dataset_name=self.train_dataset)

        # EVAL-DATASET
        for eval_dataset in self.test_datasets:
            BEIRDownloader(data_dir=self.beir_data_dir, dataset_name=eval_dataset)

    def _try_load_qrels(self, qrels_split, dataset_name):
        try:
            train_qrels = GenericDataLoader(f"{self.beir_data_dir}/{dataset_name}").load_custom(
                split=qrels_split, which="qrels"
            )
        except ValueError:
            train_qrels = {}
        return train_qrels

    def _prepare_data(
        self,
        corpus: Dict = None,
        queries: Dict = None,
        split: str = None,
        dataset_name: str = None,
        msmarco_query_count: Union[Counter, Dict] = None,
    ):
        if split is None:
            raise ValueError("set split: `train`, `test`, `dev`")

        if dataset_name is None:
            raise ValueError("specify the dataset name")

        # loading corpus, queries and qrels
        if corpus is None and queries is None:
            corpus, queries, qrels = GenericDataLoader(
                f"{self.beir_data_dir}/{dataset_name}"
            ).load(split=split)
        else:
            if split != "test":
                qrels = self._try_load_qrels(split, dataset_name)
            else:
                qrels = {
                    **self._try_load_qrels("train", dataset_name),
                    **self._try_load_qrels("dev", dataset_name),
                    **self._try_load_qrels("test", dataset_name),
                }

        return BEIRDataset(
            queries=queries,
            corpus=corpus,
            qrels=qrels,
            training=True if split == "train" else False,
        )

    def setup(self, stage: Optional[str] = None) -> None:
        """this method setups dataset on every gpu if ddp is enabled.

        :param stage: ["fit", "validate", "test", "predict"]
        """
        if stage not in ["fit", "validate", "test", "predict"]:
            raise ValueError("stage should be either `fit`, `test`, `predict`.")

        train_corpus, train_queries = None, None
        if stage in ["fit", "validate"]:
            train_dataloader = GenericDataLoader(f"{self.beir_data_dir}/{self.train_dataset}")
            train_corpus, train_queries = train_dataloader.load_custom(
                "corpus"
            ), train_dataloader.load_custom("queries")

        if stage == "fit":
            self.train_data = self._prepare_data(
                corpus=train_corpus,
                queries=train_queries,
                dataset_name="msmarco",
                split="train",
                msmarco_query_count={},
            )
            self.val_data = self._prepare_data(
                corpus=train_corpus,
                queries=train_queries,
                dataset_name="msmarco",
                split="dev",
                msmarco_query_count={},
            )
        if stage == "validate":
            self.val_data = self._prepare_data(
                corpus=train_corpus, queries=train_queries, dataset_name="msmarco", split="dev"
            )
        if stage == "test":
            self.test_data = {
                test_dataset: self._prepare_data(
                    split="test", dataset_name=test_dataset, msmarco_query_count=None
                )
                for test_dataset in self.test_datasets
            }

    def smart_batching_collate(self, batch: List[InputData]):
        """Transforms a batch from a SmartBatchingDataset to a batch of tensors for the model Here,
        batch is a list of tuples: [(tokens, label), ...]

        :param batch:
            a batch from a SmartBatchingDataset
        :return:
            a batch of tensors for the model
        """
        queries: List[str] = []
        docs: List[str] = []
        labels: List[int] = []

        for example in batch:
            queries.append(example.query)
            docs.append(example.doc)
            labels.append(example.label)

        query_tokenized: Dict[str, Union[Tensor, any]] = self.tokenizer.tokenize_query(queries)
        doc_tokenized: Dict[str, Union[Tensor, any]] = self.tokenizer.tokenize_passage(docs)

        return query_tokenized, doc_tokenized, labels

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        # Here shuffle=False, since (or assuming) we have done it in the pseudo labeling
        return DataLoader(
            self.train_data,
            collate_fn=self.smart_batching_collate,
            batch_size=self.train_batch_size,
            shuffle=self.shuffle_train,
            drop_last=True,
            num_workers=self.workers,
        )

    def val_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(
            self.val_data,
            collate_fn=self.smart_batching_collate,
            batch_size=self.test_batch_size,
            shuffle=False,
            drop_last=True,
            num_workers=self.workers,
        )

    def test_dataloader(self) -> EVAL_DATALOADERS:
        # dont shuffle because this is test dataset
        return [
            DataLoader(
                self.test_data[test_dataset],
                collate_fn=self.smart_batching_collate,
                batch_size=self.test_batch_size,
                shuffle=False,
                drop_last=True,
                num_workers=self.workers,
            )
            for test_dataset in self.test_datasets
        ]

    def predict_dataloader(self) -> EVAL_DATALOADERS:
        return self.val_dataloader()

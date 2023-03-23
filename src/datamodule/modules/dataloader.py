import csv
import logging
import os
from typing import Dict, List, Tuple

import orjson
from pytorch_lightning.utilities import rank_zero_only
from tqdm.rich import tqdm

logger = logging.getLogger(__name__)


@rank_zero_only
def log_loading(loaded_obj: Dict[str, any], obj_type: str):
    print()
    logger.info(f"Loaded {len(loaded_obj)} {obj_type}.")
    print()


class GenericDataLoader:
    def __init__(
        self,
        data_folder: str = None,
        prefix: str = None,
        corpus_file: str = "corpus.jsonl",
        query_file: str = "queries.jsonl",
        qrels_folder: str = "qrels",
        qrels_file: str = "",
    ):
        self.corpus = {}
        self.queries = {}
        self.qrels = {}

        if prefix:
            query_file = prefix + "-" + query_file
            qrels_folder = prefix + "-" + qrels_folder

        self.corpus_file = os.path.join(data_folder, corpus_file) if data_folder else corpus_file
        self.query_file = os.path.join(data_folder, query_file) if data_folder else query_file
        self.qrels_folder = os.path.join(data_folder, qrels_folder) if data_folder else None
        self.qrels_file = qrels_file

    @staticmethod
    def check(fIn: str, ext: str):
        if not os.path.exists(fIn):
            raise ValueError(f"File {fIn} not present! Please provide accurate file.")

        if not fIn.endswith(ext):
            raise ValueError(f"File {fIn} must be present with extension {ext}")

    def load_custom(
        self,
        which: str,
        split: str = None,
    ) -> Dict[str, str]:
        if which == "corpus":
            self.check(fIn=self.corpus_file, ext="jsonl")
            if not len(self.corpus):
                self._load_corpus()
            return self.corpus

        elif which == "queries":
            self.check(fIn=self.query_file, ext="jsonl")
            if not len(self.queries):
                self._load_queries()
            return self.queries

        elif which == "qrels":
            if split is None:
                raise ValueError("specify split")
            self.qrels_file = os.path.join(self.qrels_folder, split + ".tsv")
            self.check(fIn=self.qrels_file, ext="tsv")
            if os.path.exists(self.qrels_file):
                self._load_qrels()
            return self.qrels
        else:
            raise ValueError("which must be either `corpus`, `queries`, `qrels`")

    def load(
        self, split="test"
    ) -> Tuple[Dict[str, Dict[str, str]], Dict[str, str], Dict[str, Dict[str, int]]]:
        self.qrels_file = os.path.join(self.qrels_folder, split + ".tsv")
        self.check(fIn=self.corpus_file, ext="jsonl")
        self.check(fIn=self.query_file, ext="jsonl")
        self.check(fIn=self.qrels_file, ext="tsv")

        if not len(self.corpus):
            self._load_corpus()

        if not len(self.queries):
            self._load_queries()

        if os.path.exists(self.qrels_file):
            self._load_qrels()
            self.queries = {qid: self.queries[qid] for qid in self.qrels}

        return self.corpus, self.queries, self.qrels

    def load_corpus(self) -> Dict[str, Dict[str, str]]:
        self.check(fIn=self.corpus_file, ext="jsonl")

        if not len(self.corpus):
            self._load_corpus()

        return self.corpus

    def _load_corpus(self):
        num_lines = sum(1 for i in open(self.corpus_file, "rb"))
        with open(self.corpus_file, encoding="utf8") as fIn:
            for line in tqdm(
                fIn,
                total=num_lines,
                desc="Loading Corpus",
                unit_scale=1000000,
            ):
                line = orjson.loads(line)
                self.corpus[line.get("_id")] = {
                    "text": line.get("text"),
                    "title": line.get("title"),
                }

    def _load_queries(self):
        num_lines = sum(1 for i in open(self.query_file, "rb"))
        with open(self.query_file, encoding="utf8") as fIn:
            for line in tqdm(
                fIn,
                total=num_lines,
                desc="Loading Queries",
                unit_scale=1000000000,
            ):
                line = orjson.loads(line)
                self.queries[line.get("_id")] = line.get("text")

    def _load_qrels(self):
        reader = csv.reader(
            open(self.qrels_file, encoding="utf-8"), delimiter="\t", quoting=csv.QUOTE_MINIMAL
        )
        next(reader)

        for id, row in enumerate(reader):
            query_id, corpus_id, score = row[0], row[1], int(row[2])

            if query_id not in self.qrels:
                self.qrels[query_id] = {corpus_id: score}
            else:
                self.qrels[query_id][corpus_id] = score

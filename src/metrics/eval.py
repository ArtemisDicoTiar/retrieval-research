from itertools import chain
from typing import Dict

import torch
from beir.retrieval.evaluation import EvaluateRetrieval
from einops import rearrange
from torch import Tensor
from torchmetrics import Metric


class EvaluationMetric(Metric):
    def __iter__(self):
        pass

    full_state_update = False

    def __init__(self, k_values=None):
        super().__init__()
        if k_values is None:
            k_values = [1, 3, 5, 10, 20, 100]

        self.k_values = k_values
        self.add_state("idxs", default=torch.empty(0), dist_reduce_fx="cat")
        self.add_state("labels", default=torch.empty(0), dist_reduce_fx="cat")
        self.add_state("preds", default=torch.empty(0), dist_reduce_fx="cat")
        self.add_state("reranks", default=torch.empty(0), dist_reduce_fx="cat")

    def update(self, index: Tensor, pred: Tensor, label: Tensor, rerank: Tensor):
        self.idxs = torch.cat((self.idxs, index), dim=0) if self.idxs.shape[0] != 0 else index
        self.labels = (
            torch.cat((self.labels, label), dim=0) if self.labels.shape[0] != 0 else label
        )
        self.preds = torch.cat((self.preds, pred), dim=0) if self.preds.shape[0] != 0 else pred
        self.reranks = (
            torch.cat((self.reranks, rerank), dim=0) if self.reranks.shape[0] != 0 else rerank
        )

    def compute(self) -> Dict[str, Dict[str, float]]:
        full_results = {}
        reranks: Dict[str, Dict[str, any]] = {}
        qrels = {}
        if len(self.idxs.shape) == 3:
            self.idxs = rearrange(self.idxs, "gpus batch ids -> (gpus batch) ids")
            self.preds = rearrange(self.preds, "gpus batch -> (gpus batch)")
            self.labels = rearrange(self.labels, "gpus batch -> (gpus batch)")
            self.reranks = rearrange(self.reranks, "gpus batch -> (gpus batch)")

        for i, idx in enumerate(self.idxs):
            qid, pid = idx
            qid = str(qid.item())
            pid = str(pid.item())

            if qid not in full_results:
                full_results[qid] = {}

            if qid not in reranks:
                reranks[qid] = {}

            if qid not in qrels:
                qrels[qid] = {}

            full_results[qid][pid] = self.preds[i].item()
            reranks[qid][pid] = self.reranks[i].item()
            qrels[qid][pid] = self.labels[i].item()

        re_rank_targets = list(
            filter(
                lambda item: item is not None,
                chain.from_iterable(
                    map(
                        lambda q: map(
                            lambda p: (q, p) if reranks[q][p] == 1 else None,
                            reranks[q].keys(),
                        ),
                        reranks.keys(),
                    )
                ),
            )
        )

        re_qrels = {}
        re_results = {}
        for target_qid, target_pid in re_rank_targets:
            if target_qid not in re_results:
                re_results[target_qid] = {}
                re_qrels[target_qid] = {}

            re_results[target_qid][target_pid] = full_results[target_qid][target_pid]
            re_qrels[target_qid][target_pid] = qrels[target_qid][target_pid]

        full_ndcg, full_map, full_recall, full_precision = EvaluateRetrieval.evaluate(
            qrels, full_results, k_values=self.k_values
        )
        full_mrr = EvaluateRetrieval.evaluate_custom(
            qrels, full_results, k_values=self.k_values, metric="mrr"
        )

        re_ndcg, re_map, re_recall, re_precision, re_mrr = -1, -1, -1, -1, -1
        if re_results:
            re_ndcg, re_map, re_recall, re_precision = EvaluateRetrieval.evaluate(
                re_qrels, re_results, k_values=self.k_values
            )
            re_mrr = EvaluateRetrieval.evaluate_custom(
                re_qrels, re_results, k_values=self.k_values, metric="mrr"
            )

        return {
            # full-rank
            "fl.ndcg": full_ndcg,
            "fl.map": full_map,
            "fl.recall": full_recall,
            "fl.precision": full_precision,
            "fl.mrr": full_mrr,
            # re-rank
            "re.ndcg": re_ndcg,
            "re.map": re_map,
            "re.recall": re_recall,
            "re.precision": re_precision,
            "re.mrr": re_mrr,
        }

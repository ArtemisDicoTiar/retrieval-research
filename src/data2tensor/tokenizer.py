import string
from typing import Dict, List, Union

import torch
from torch import Tensor
from transformers import AutoTokenizer, BatchEncoding, BertTokenizer, T5Tokenizer


class BaseTokenizer:
    def __init__(self, model_name_or_path: str, max_query_length: int, max_doc_length: int):
        self.max_query_length = max_query_length
        self.max_doc_length = max_doc_length

        self.tokenizer: BertTokenizer = BertTokenizer.from_pretrained(model_name_or_path)

        # self.skiplist = {
        #     w: True
        #     for symbol in string.punctuation
        #     for w in [symbol, self.tokenizer.encode(symbol, add_special_tokens=False)[0]]
        # }

    def make_human_readable(self, tensor: Tensor):
        return self.tokenizer.batch_decode(tensor)

    def padding_expansion(self, encoded_tokens: BatchEncoding):
        # sep -> mask, pad -> mask
        encoded_tokens["input_ids"][
            encoded_tokens["input_ids"] == self.tokenizer.sep_token_id
        ] = self.tokenizer.mask_token_id
        encoded_tokens["input_ids"][
            encoded_tokens["input_ids"] == self.tokenizer.pad_token_id
        ] = self.tokenizer.mask_token_id
        # last token -> sep
        encoded_tokens["input_ids"][:, -1] = self.tokenizer.sep_token_id

        # make all tokens to be activated.
        encoded_tokens["attention_mask"] = 1

        return encoded_tokens

    def tokenize_query(self, query: List[str]) -> Union[BatchEncoding, Dict[str, Tensor]]:
        query = self.tokenizer(
            query,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
            max_length=self.max_query_length,
        )
        expanded_query = self.padding_expansion(query)
        return expanded_query

    def tokenize_passage(self, passage: List[str]) -> Union[BatchEncoding, Dict[str, Tensor]]:
        passage = self.tokenizer(
            passage,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
            max_length=self.max_doc_length,
        )
        return passage

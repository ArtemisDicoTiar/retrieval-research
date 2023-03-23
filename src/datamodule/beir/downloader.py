import os
from pathlib import Path

from src import utils
from src.datamodule.modules.downloader import DataDownloader, SSHDownloader
from src.datamodule.utils import ssh_connector

log = utils.get_pylogger(__name__)


class BEIRDownloader(DataDownloader):
    def __init__(self, dataset_name: str, data_dir: str):
        super().__init__(
            "beir",
            dataset_name,
            data_dir,
            data_essentials=["corpus.jsonl", "queries.jsonl", "qrels"],
        )

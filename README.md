# retrieval research base repo

[![python](https://img.shields.io/badge/-Python_3.8_%7C_3.9_%7C_3.10-blue?logo=python&logoColor=white)](https://github.com/pre-commit/pre-commit)
[![pytorch](<https://img.shields.io/badge/PyTorch_1.11+(cu11.3)-ee4c2c?logo=pytorch&logoColor=white>)](https://pytorch.org/get-started/locally/)
[![lightning](https://img.shields.io/badge/-Lightning_1.7.7+-792ee5?logo=pytorchlightning&logoColor=white)](https://pytorchlightning.ai/)
[![hydra](https://img.shields.io/badge/Config-Hydra_1.2-89b8cd)](https://hydra.cc/)
[![black](https://img.shields.io/badge/Code%20Style-Black-black.svg?labelColor=gray)](https://black.readthedocs.io/en/stable/)

[![Code Quality Main](https://github.com/ArtemisDicoTiar/distilColBERT/actions/workflows/code-quality-main.yaml/badge.svg)](https://github.com/ArtemisDicoTiar/distilColBERT/actions/workflows/code-quality-main.yaml)
[![Tests](https://github.com/ArtemisDicoTiar/distilColBERT/actions/workflows/test.yaml/badge.svg)](https://github.com/ArtemisDicoTiar/distilColBERT/actions/workflows/test.yaml)
[![codecov](https://codecov.io/github/ArtemisDicoTiar/distilColBERT/branch/main/graph/badge.svg?token=IO3IIQXF71)](https://codecov.io/github/ArtemisDicoTiar/distilColBERT)

## 📌  Introduction

### Research Topic

### [🔬 Experiment status visible on 🔗 W&B](<>)

## Project Structure

The directory structure of new project looks like this:

```
├── configs                   <- Hydra configuration files
│   ├── callbacks                <- Callbacks configs
│   ├── datamodule               <- Datamodule configs
│   ├── experiment               <- Experiment configs
│   ├── extras                   <- Extra utilities configs
│   ├── hparams_search           <- Hyperparameter search configs
│   ├── hydra                    <- Hydra configs
│   ├── local                    <- Local configs
│   ├── logger                   <- Logger configs
│   ├── model                    <- Model configs
│   ├── paths                    <- Project paths configs
│   ├── tokenizer                <- Tokenizer configs
│   ├── trainer                  <- Trainer configs
│   │
│   ├── eval.yaml             <- Main config for evaluation
│   └── train.yaml            <- Main config for training
│
├── data                   <- Project data (will be generated automatically)
│
├── logs                   <- Logs generated by hydra and lightning loggers
│
├── scripts                <- Shell scripts
│
├── distilColBERT                <- Source code
│   ├── data2tensor              <- Tokenizers
│   ├── datamodule               <- Lightning datamodules
│   ├── losses                   <- Loss functions
│   ├── metrics                  <- torchmetrics based metrics
│   ├── models                   <- Lightning models
│   ├── utils                    <- Utility scripts
│   │
│   ├── eval.py                  <- Run evaluation
│   └── train.py                 <- Run training
│
├── .gitignore                <- List of files ignored by git
├── pyproject.toml            <- Configuration options for testing and linting
├── requirements.txt          <- File for installing python dependencies
└── README.md
```

## 🚀  Quickstart

```bash
# clone project
git clone
cd distilColBERT

# [OPTIONAL] create conda environment
conda create -n myenv python=3.9
conda activate myenv

# install pytorch according to instructions
# https://pytorch.org/get-started/

# install requirements
pip install -r requirements.txt

# run training
python train.py

# run evaluation
python eval.py
```

## 💡 How to

### 👓 Read Code

This repository is consist of mainly two parts, `configs` and `src (the main source code)`
.

- Configs
  - Configs are mostly written beforehand. Therefore, some yaml files may be difficult to understand.
  - Only config folder that you need to understand is `experiment`, `model` and `tokenizer`
    - `experiment` folder includes experiment settings
    - `model` folder contains Pytorch-Lightning module configuration (eg. retrieval_module)
    - `tokenizer` folder shows the tokenizer setting. By default, it is set as `BaseTokenizer`.
- distilColBERT (source code)
  - `data2tensor` defines how data (text) is being tokenized with huggingface tokenizer.
  - `datamodule` defines pytorch-lightning datamodule which loading GPL data (140k steps, 32 batch) for training and BEIR data for evaluation.
  - `losses` define loss functions (objectives).
    - `modules` are custom sub-functions for loss function (eg. MarginMSELoss, KL-Divergence)
  - `metrics` define metrics for either train or evaluation. (evaluation metrics written in advance)
  - `model` defines retrieval model.
    - `modules` are layers used by retrieval model.
  - `utils` is for logging, wrappers and hydra instantiation written from template.

### 🖋 Write code (workflow)

If you are trying to experiment new model, follow these steps.

1. define your model in `src.model` (if new sub-layers are needed, write in `modules`).
2. write the objectives in `src.losses` (if new sub-functions are needed, write in `modules`).
3. describe your setting in `configs.model` with appropriate name.
   - This name is used on experiment configs
   - if help needed
     - try to read `default.yaml` for which key-values are set.
     - try to read `colbert.yaml` as an example.
4. illustrate your experiment configuration on `configs.experiment` with relevant name.
   - This name is used on running train and eval.
   - if help needed try to read `example.yaml`.
5. run train: `python train.py experiment={experiment name} datamodule.data_name={beir dataset name you want to use}`
6. run eval:  `python eval.py experiment={experiment name} datamodule.data_name={beir dataset name} ckpt_path={checkpoint path}`

### 🪄 Magics for running train/eval (from template)

<details>
<summary><b>Override any config parameter from command line</b></summary>

```bash
python train.py trainer.max_epochs=20 model.optimizer.lr=1e-4
```

> **Note**: You can also add new parameters with `+` sign.

```bash
python train.py +model.new_param="owo"
```

</details>

<details>
<summary><b>Train on CPU, GPU, multi-GPU and TPU</b></summary>

```bash
# train on CPU
python train.py trainer=cpu

# train on 1 GPU
python train.py trainer=gpu

# train on TPU
python train.py +trainer.tpu_cores=8

# train with DDP (Distributed Data Parallel) (4 GPUs)
python train.py trainer=ddp trainer.devices=4

# train with DDP (Distributed Data Parallel) (8 GPUs, 2 nodes)
python train.py trainer=ddp trainer.devices=4 trainer.num_nodes=2

# simulate DDP on CPU processes
python train.py trainer=ddp_sim trainer.devices=2

# accelerate training on mac
python train.py trainer=mps
```

> **Warning**: Currently there are problems with DDP mode, read [this issue](https://github.com/ashleve/lightning-hydra-template/issues/393) to learn more.

</details>

<details>
<summary><b>Train with mixed precision</b></summary>

```bash
# train with pytorch native automatic mixed precision (AMP)
python train.py trainer=gpu +trainer.precision=16
```

</details>

<details>
<summary><b>Optimize large scale models on multiple GPUs with Deepspeed</b></summary>

```bash
python train.py +trainer.
```

</details>

<details>
<summary><b>Train model with any logger available in PyTorch Lightning, like W&B or Tensorboard</b></summary>

```yaml
# set project and entity names in `configs/logger/wandb`
wandb:
  project: "your_project_name"
  entity: "your_wandb_team_name"
```

```bash
# train model with Weights&Biases (link to wandb dashboard should appear in the terminal)
python train.py logger=wandb
```

> **Note**: Lightning provides convenient integrations with most popular logging frameworks. Learn more [here](#experiment-tracking).

> **Note**: Using wandb requires you to [setup account](https://www.wandb.com/) first. After that just complete the config as below.

> **Note**: Click [here](https://wandb.ai/hobglob/template-dashboard/) to see example wandb dashboard generated with this template.

</details>

<details>
<summary><b>Train model with chosen experiment config</b></summary>

```bash
python train.py experiment=example
```

> **Note**: Experiment configs are placed in [configs/experiment/](configs/experiment/).

</details>

<details>
<summary><b>Attach some callbacks to run</b></summary>

```bash
python train.py callbacks=default
```

> **Note**: Callbacks can be used for things such as as model checkpointing, early stopping and [many more](https://pytorch-lightning.readthedocs.io/en/latest/extensions/callbacks.html#built-in-callbacks).

> **Note**: Callbacks configs are placed in [configs/callbacks/](configs/callbacks/).

</details>

<details>
<summary><b>Use different tricks available in Pytorch Lightning</b></summary>

```yaml
# gradient clipping may be enabled to avoid exploding gradients
python train.py +trainer.gradient_clip_val=0.5

# run validation loop 4 times during a training epoch
python train.py +trainer.val_check_interval=0.25

# accumulate gradients
python train.py +trainer.accumulate_grad_batches=10

# terminate training after 12 hours
python train.py +trainer.max_time="00:12:00:00"
```

> **Note**: PyTorch Lightning provides about [40+ useful trainer flags](https://pytorch-lightning.readthedocs.io/en/latest/common/trainer.html#trainer-flags).

</details>

<details>
<summary><b>Easily debug</b></summary>

```bash
# runs 1 epoch in default debugging mode
# changes logging directory to `logs/debugs/...`
# sets level of all command line loggers to 'DEBUG'
# enforces debug-friendly configuration
python train.py debug=default

# run 1 train, val and test loop, using only 1 batch
python train.py debug=fdr

# print execution time profiling
python train.py debug=profiler

# try overfitting to 1 batch
python train.py debug=overfit

# raise exception if there are any numerical anomalies in tensors, like NaN or +/-inf
python train.py +trainer.detect_anomaly=true

# log second gradient norm of the model
python train.py +trainer.track_grad_norm=2

# use only 20% of the data
python train.py +trainer.limit_train_batches=0.2 \
+trainer.limit_val_batches=0.2 +trainer.limit_test_batches=0.2
```

> **Note**: Visit [configs/debug/](configs/debug/) for different debugging configs.

</details>

<details>
<summary><b>Resume training from checkpoint</b></summary>

```yaml
python train.py ckpt_path="/path/to/ckpt/name.ckpt"
```

> **Note**: Checkpoint can be either path or URL.

> **Note**: Currently loading ckpt doesn't resume logger experiment, but it will be supported in future Lightning release.

</details>

<details>
<summary><b>Evaluate checkpoint on test dataset</b></summary>

```yaml
python eval.py ckpt_path="/path/to/ckpt/name.ckpt"
```

> **Note**: Checkpoint can be either path or URL.

</details>

<details>
<summary><b>Create a sweep over hyperparameters</b></summary>

```bash
# this will run 6 experiments one after the other,
# each with different combination of batch_size and learning rate
python train.py -m datamodule.batch_size=32,64,128 model.lr=0.001,0.0005
```

> **Note**: Hydra composes configs lazily at job launch time. If you change code or configs after launching a job/sweep, the final composed configs might be impacted.

</details>

<details>
<summary><b>Create a sweep over hyperparameters with Optuna</b></summary>

```bash
# this will run hyperparameter search defined in `configs/hparams_search/example.yaml`
# over chosen experiment config
python train.py -m hparams_search=mnist_optuna experiment=example
```

> **Note**: Using [Optuna Sweeper](https://hydra.cc/docs/next/plugins/optuna_sweeper) doesn't require you to add any boilerplate to your code, everything is defined in a [single config file](configs/hparams_search/example.yaml).

> **Warning**: Optuna sweeps are not failure-resistant (if one job crashes then the whole sweep crashes).

</details>

<details>
<summary><b>Execute all experiments from folder</b></summary>

```bash
python train.py -m 'experiment=glob(*)'
```

> **Note**: Hydra provides special syntax for controlling behavior of multiruns. Learn more [here](https://hydra.cc/docs/next/tutorials/basic/running_your_app/multi-run). The command above executes all experiments from [configs/experiment/](configs/experiment/).

</details>

<details>
<summary><b>Execute run for multiple different seeds</b></summary>

```bash
python train.py -m seed=1,2,3,4,5 trainer.deterministic=True logger=csv tags=["benchmark"]
```

> **Note**: `trainer.deterministic=True` makes pytorch more deterministic but impacts the performance.

</details>

<details>
<summary><b>Use Hydra tab completion</b></summary>

> **Note**: Hydra allows you to autocomplete config argument overrides in shell as you write them, by pressing `tab` key. Read the [docs](https://hydra.cc/docs/tutorials/basic/running_your_app/tab_completion).

```bash
eval "$(python train.py -sc install=bash)"
eval "$(python eval.py -sc install=bash)"
```

</details>

<details>
<summary><b>Apply pre-commit hooks</b></summary>

```bash
pre-commit run -a
```

> **Note**: Apply pre-commit hooks to do things like auto-formatting code and configs, performing code analysis or removing output from jupyter notebooks. See [# Best Practices](#best-practices) for more.

</details>

<details>
<summary><b>Use tags</b></summary>

Each experiment should be tagged in order to easily filter them across files or in logger UI:

```bash
python train.py tags=["mnist", "experiment_X"]
```

If no tags are provided, you will be asked to input them from command line:

```bash
>>> python train.py tags=[]
[2022-07-11 15:40:09,358][src.utils.utils][INFO] - Enforcing tags! <cfg.extras.enforce_tags=True>
[2022-07-11 15:40:09,359][src.utils.rich_utils][WARNING] - No tags provided in config. Prompting user to input tags...
Enter a list of comma separated tags (dev):
```

If no tags are provided for multirun, an error will be raised:

```bash
>>> python train.py -m +x=1,2,3 tags=[]
ValueError: Specify tags before launching a multirun!
```

> **Note**: Appending lists from command line is currently not supported in hydra :(

</details>

<br>

## 📚  References

- Repo Template
  - Thanks to "ashleve" providing [this template](https://github.com/ashleve/lightning-hydra-template)
- Weight and Bias (W&B)
  ```
  @misc{wandb,
  title =        {Experiment Tracking with Weights and Biases},
  year =         {2020},
  note =         {Software available from wandb.com},
  url=           {https://www.wandb.com/},
  author =       {Biewald, Lukas},
  }
  ```
- Hydra
  ```
  @Misc{Yadan2019Hydra,
  author =       {Omry Yadan},
  title =        {Hydra - A framework for elegantly configuring complex applications},
  howpublished = {Github},
  year =         {2019},
  url =          {https://github.com/facebookresearch/hydra}
  }

  ```
- ColBERT
  ```
  @article{DBLP:journals/corr/abs-2004-12832,
  author    = {Omar Khattab and
               Matei Zaharia},
  title     = {ColBERT: Efficient and Effective Passage Search via Contextualized
               Late Interaction over {BERT}},
  journal   = {CoRR},
  volume    = {abs/2004.12832},
  year      = {2020},
  url       = {https://arxiv.org/abs/2004.12832},
  eprinttype = {arXiv},
  eprint    = {2004.12832},
  timestamp = {Wed, 29 Apr 2020 10:17:11 +0200},
  biburl    = {https://dblp.org/rec/journals/corr/abs-2004-12832.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
  }
  ```
- GPL
  ```
  @article{wang2021gpl,
    title = "GPL: Generative Pseudo Labeling for Unsupervised Domain Adaptation of Dense Retrieval",
    author = "Kexin Wang and Nandan Thakur and Nils Reimers and Iryna Gurevych",
    journal= "arXiv preprint arXiv:2112.07577",
    month = "4",
    year = "2021",
    url = "https://arxiv.org/abs/2112.07577",
  }
  ```
- BEIR
  ```
  @inproceedings{
    thakur2021beir,
    title={{BEIR}: A Heterogeneous Benchmark for Zero-shot Evaluation of Information Retrieval Models},
    author={Nandan Thakur and Nils Reimers and Andreas R{\"u}ckl{\'e} and Abhishek Srivastava and Iryna Gurevych},
    booktitle={Thirty-fifth Conference on Neural Information Processing Systems Datasets and Benchmarks Track (Round 2)},
    year={2021},
    url={https://openreview.net/forum?id=wCu6T5xFjeJ}
  }
  ```
- huggingface
  ```
  @inproceedings{wolf-etal-2020-transformers,
      title = "Transformers: State-of-the-Art Natural Language Processing",
      author = "Thomas Wolf and Lysandre Debut and Victor Sanh and Julien Chaumond and Clement Delangue and Anthony Moi and Pierric Cistac and Tim Rault and Rémi Louf and Morgan Funtowicz and Joe Davison and Sam Shleifer and Patrick von Platen and Clara Ma and Yacine Jernite and Julien Plu and Canwen Xu and Teven Le Scao and Sylvain Gugger and Mariama Drame and Quentin Lhoest and Alexander M. Rush",
      booktitle = "Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing: System Demonstrations",
      month = oct,
      year = "2020",
      address = "Online",
      publisher = "Association for Computational Linguistics",
      url = "https://www.aclweb.org/anthology/2020.emnlp-demos.6",
      pages = "38--45"
  }
  ```
- SBERT
  ```
  @inproceedings{reimers-2019-sentence-bert,
    title = "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks",
    author = "Reimers, Nils and Gurevych, Iryna",
    booktitle = "Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing",
    month = "11",
    year = "2019",
    publisher = "Association for Computational Linguistics",
    url = "https://arxiv.org/abs/1908.10084",
  }
  ```

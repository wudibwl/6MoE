# 6MoE

6MoE: A Mixture-of-Experts Transformer for Heterogeneous IPv6 Target Generation
---

## Requirements & Prerequisites

Make sure you have installed the required dependencies:

```bash
pip install torch tqdm numpy ordered-set
```

---

## Standard Usage Modes

### 1. Training a Model from Scratch
To train the model on a seed dataset (`--no_train` must be overridden, as generation is default):

```bash
python run6MoE.py --seed_file data/filtered_ipv6_32hex.txt \
                  --model_file data/run6MoE.pth \
                  --epochs 50 \
                  --batch_size 64 \
                  --learning_rate 0.0005
```

### 2. Generating Candidate IPv6 Addresses
By default, `--no_train` is enabled. It will load model weights from `--model_file` and output generated address candidates allocated proportionally based on `/64` prefix frequencies found in your `--seed_file`.

```bash
python run6MoE.py --model_file data/run6MoE.pth \
                  --seed_file data/filtered_ipv6_32hex.txt \
                  --candidate_file data/candidates.txt \
                  --budget 100000 \
                  --temperature 0.8 \
                  --top_k 16
```

### 3. Fine-Tuning an Existing Model
To load pretrained weights and perform fine-tuning using active/alive IPv6 addresses (e.g., discovered via scanning):

```bash
python run6MoE.py --finetune \
                  --model_file data/run6MoE.pth \
                  --alive_file data/alive_ipv6.txt \
                  --finetune_epochs 5 \
                  --batch_size 64
```

---

## Command Line Arguments Reference

| Argument | Type | Default Value | Description |
| :--- | :--- | :--- | :--- |
| `--seed_file` | `str` | `data/filtered_ipv6_32hex.txt` | Path to raw/seed IPv6 dataset (32 hex nibbles). |
| `--model_file` | `str` | `data/run6MoE.pth` | Path to save/load model weights `.pth`. |
| `--candidate_file` | `str` | `data/candidates.txt` | Target output path for generated IPv6 addresses. |
| `--alive_file` | `str` | `data/` | Dataset path containing verified alive addresses for fine-tuning. |
| `--budget` | `int` | `100000` | Target total number of candidate addresses to generate. |
| `--epochs` | `int` | `50` | Number of epochs for standard model training. |
| `--finetune_epochs` | `int` | `5` | Epochs used when running `--finetune`. |
| `--batch_size` | `int` | `64` | Batch size for training and sampling routines. |
| `--learning_rate` | `float` | `0.0005` | Learning rate for Adam optimizer. |
| `--temperature` | `float` | `0.8` | Softmax temperature scaling for generation sampling diversity. |
| `--top_k` | `int` | `16` | Top-$k$ filtering bound for next-token sampling ($\le 16$). |
| `--finetune` | `flag` | `False` | Enables fine-tuning mode on the dataset specified in `--alive_file`. |

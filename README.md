# GRPO — Uncertainty Decomposition Training

GRPO (Group Relative Policy Optimization) training with question-parallel DDP.
The policy is trained to predict a **total / share** uncertainty decomposition for
a model's own answer to a question:

- `Total_uncertainty` → P(answer is wrong)
- `Aleatoric_share` → fraction of that uncertainty caused by question ambiguity

from which `aleatoric = share * total` and `epistemic = (1 - share) * total`.
Targets are the calibrated fields, which satisfy `aleatoric + epistemic = total`.

## Layout

```
GRPO/
├── grpo_ddp.py          # training script
├── slurm_ddp.sh         # SLURM launcher (torchrun, 8 GPUs)
├── .env                 # HF_HOME / HF_HUB_CACHE / WANDB_API_KEY
├── utils/
│   ├── grpo.py          # generate_grpo_samples_batched, compute_batched_log_probs
│   └── prompts.py       # TOTAL_SHARE_GIVEN_ANSWER_PROMPT
├── data/scaled/AmbigQA/isotonic/
│   └── qwen_3.5_9b_all_temp1.0.json     # training/eval dataset (calibrated targets)
└── models/qwen_3.5_9b/sft/wo_correctness/run_1782808950/   # SFT warm-start adapter
```

## Requirements

- Python env with: `torch`, `transformers`, `peft`, `wandb`, `scipy`, `numpy`,
  `python-dotenv`.
- The base model `Qwen/Qwen3.5-9B` available in the HuggingFace cache pointed to by
  `HF_HOME` / `HF_HUB_CACHE` in `.env` (not shipped with this folder).
- 8 GPUs (configurable — see below).
- Copy `.env.example` to `.env` and fill in your keys/paths.
- The SFT warm-start weights (`models/.../run_1782808950/adapter_model.safetensors`,
  ~166 MB) are **not tracked in git** (over GitHub's file-size limit). Supply them
  separately, or set `INIT_FROM_SFT = False` in `grpo_ddp.py` to train from scratch.

## Run

On SLURM:

```bash
cd GRPO
sbatch slurm_ddp.sh
```

Directly with torchrun:

```bash
cd GRPO
source /path/to/venv/bin/activate
export $(grep -v '^#' .env | xargs)
torchrun --nproc_per_node=8 --master_port=29500 grpo_ddp.py --error_type absolute
```

`--error_type` selects the reward shaping: `mse`, `exp_abs`, or `absolute`.

## Key configuration

Set at the top of `grpo_ddp.py`:

| Setting | Default | Description |
|---|---|---|
| `MODEL` | `Qwen/Qwen3.5-9B` | Base model |
| `DATASET_PATH` | `data/scaled/AmbigQA/isotonic/qwen_3.5_9b_all_temp1.0.json` | Training data |
| `BATCH_SIZE` | 72 | Questions per step across all GPUs |
| `NUM_SAMPLES_PER_PROMPT` | 16 | Rollouts per question (group size) |
| `TOTAL_STEPS` | 350 | Training steps |
| `LEARNING_RATE` | 7e-6 | Cosine-decayed to `LR_MIN` (1e-6) |
| `INIT_FROM_SFT` | True | Warm-start from `SFT_ADAPTER_PATH` |
| `USE_KL_PENALTY` / `KL_BETA` | True / 0.04 | KL anchor vs adapter-disabled base |
| `STRATIFIED_SAMPLING` | True | 9-cell (aleatoric × epistemic) batch sampler |

## Outputs

Written under the run directory (created automatically):

- `models/qwen_3.5_9b/AmbigQA/single_grpo/wo_correctness/run_<ts>/` — LoRA
  checkpoints (every 100 steps) and the final adapter.
- `training_results/qwen_3.5_9b/AmbigQA/single_grpo/wo_correctness/run_<ts>/` —
  eval metrics (every 50 steps), metadata, and the test split.

Metrics are also logged to Weights & Biases (project `uncertainty_rl`).

"""
GRPO training with question-parallel DDP.

Each GPU processes BATCH_SIZE / world_size questions, computes rewards and
advantages locally, then all-reduces gradients before the optimizer step.

The policy predicts a total/share decomposition:
    Total_uncertainty  -> P(answer wrong)
    Aleatoric_share    -> fraction of total caused by ambiguity
with aleatoric = share * total and epistemic = (1 - share) * total.
Targets are the calibrated fields, which satisfy aleatoric + epistemic = total.

Launch:
    torchrun --nproc_per_node=NUM_GPUS grpo_ddp.py --error_type "exp_abs"
"""
import argparse
import json
import random
import os
import torch
import torch.distributed as dist
import re
import numpy as np
import wandb
from utils.grpo import generate_grpo_samples_batched, compute_batched_log_probs
from utils.prompts import TOTAL_SHARE_GIVEN_ANSWER_PROMPT
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from peft import LoraConfig, get_peft_model, PeftModel, TaskType
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from dotenv import load_dotenv
from scipy.stats import pearsonr, spearmanr, rankdata
import time

load_dotenv()
os.environ["HF_HOME"] = os.getenv("HF_HOME", "")
os.environ["HF_HUB_CACHE"] = os.getenv("HF_HUB_CACHE", "")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MODEL = "Qwen/Qwen3.5-9B"
MODEL_NAME = "qwen_3.5_9b"
dataset = "AmbigQA"
THINKING = True
DATASET_PATH = f"data/scaled/{dataset}/isotonic/{MODEL_NAME}_all_temp1.0.json"
ENABLE_THINKING = False

# Calibrated decomposition targets: u_aleatoric + u_epistemic = u_total.
GT_ALEATORIC_FIELD = "u_aleatoric_calibrated"
GT_EPISTEMIC_FIELD = "u_epistemic_calibrated"
GT_TOTAL_FIELD = "u_total_calibrated"

NUM_SAMPLES_PER_PROMPT = 16
BACKWARD_MICRO_BATCH_SIZE = 2
BATCH_SIZE = 72                  # total across all GPUs; multiple of world_size and of 9 (sampler)
GRAD_ACCUM_STEPS = 1
TOTAL_STEPS = 350
TRAINING_TEMPERATURE = 1.2
EVAL_TEMPERATURE = 0.7
MAX_NEW_TOKENS = 200             # generation cap (stable gen_len is ~50-70)

REWARD_SCALING_FACTOR = 2.5      # final exp_abs scale
REWARD_SCALE_START = 1.5         # initial exp_abs scale
REWARD_SCALE_WARMUP_STEPS = 200
STD_FLOOR = 0.05                 # min group std for advantage normalisation
RATIO_MIN_TOTAL = 0.05           # skip the share objective below this gt total
TOTAL_ADV_WEIGHT = 1.0
SHARE_ADV_WEIGHT = 1.0
INTERP_ADV_WEIGHT = 1.0          # weight on distinct-readings vs num_clarifications reward
INTERP_CAP = 5                   # saturate count match at this many readings
QUADRANT_THRESHOLD = 0.25
QUADRANT_HIGH_THRESHOLD = 0.5
LORA_DROPOUT = 0.0

INIT_FROM_SFT = True
SFT_ADAPTER_PATH = "models/qwen_3.5_9b/sft/wo_correctness/run_1782808950"

USE_KL_PENALTY = True
KL_BETA = 0.04                   # KL anchor vs adapter-disabled base

LEARNING_RATE = 7e-6
LR_MIN = 1e-6
USE_LR_DECAY = True

SEED = True
STRATIFIED_SAMPLING = True
WITH_CORRECTNESS_REWARD = "wo_correctness"                

model_saving_dir = f"models/{MODEL_NAME}/{dataset}/single_grpo/{WITH_CORRECTNESS_REWARD}"
results_saving_dir = f"training_results/{MODEL_NAME}/{dataset}/single_grpo/{WITH_CORRECTNESS_REWARD}"
DESCRIPTION = (
    f"GRPO question-parallel DDP on {MODEL_NAME} for {dataset}. Answer-first total/share "
    f"decomposition on calibrated targets. LR={LEARNING_RATE} cosine->{LR_MIN} over {TOTAL_STEPS} steps. "
    f"Per-axis std-normalised advantage (total weight={TOTAL_ADV_WEIGHT}, share weight={SHARE_ADV_WEIGHT}). "
    f"KL penalty {'on' if USE_KL_PENALTY else 'off'} (beta={KL_BETA}). "
    f"Stratified sampling {'on' if STRATIFIED_SAMPLING else 'off'}. SFT init {'on' if INIT_FROM_SFT else 'off'}."
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def get_system_prompt(step):
    return TOTAL_SHARE_GIVEN_ANSWER_PROMPT


def load_json_dataset_wo_balance(json_path: str, test_ratio: float = 0.05):
    with open(json_path, "r") as f:
        data = json.load(f)
    assert isinstance(data, list), "JSON file must contain a list of samples"
    random.shuffle(data)
    split_idx = int(len(data) * (1 - test_ratio))
    return data[:split_idx], data[split_idx:]


def extract_response_components(text):
    # Take the LAST occurrence of each score (reasoning may mention candidates first).
    total_matches = re.findall(r"Total_uncertainty:\s*([0-9]*\.?[0-9]+)", text)
    share_matches = re.findall(r"Aleatoric_share:\s*([0-9]*\.?[0-9]+)", text)
    if not (total_matches and share_matches):
        return None
    total = float(np.clip(float(total_matches[-1]), 0.0, 1.0))
    share = float(np.clip(float(share_matches[-1]), 0.0, 1.0))
    answer_match = re.search(r"Answer:\s*(.+)", text)
    nint_match = re.search(r"N_interpretations:\s*([0-9]+)", text)
    return {
        "total": total,
        "share": share,
        "aleatoric": share * total,
        "epistemic": (1.0 - share) * total,
        "answer": answer_match.group(1).strip() if answer_match else "",
        "n_interp": int(nint_match.group(1)) if nint_match else None,
    }


def get_scale_factor(step: int) -> float:
    if step >= REWARD_SCALE_WARMUP_STEPS:
        return REWARD_SCALING_FACTOR
    t = step / REWARD_SCALE_WARMUP_STEPS
    return REWARD_SCALE_START + t * (REWARD_SCALING_FACTOR - REWARD_SCALE_START)


# ---------------------------------------------------------------------------
# Reward
# ---------------------------------------------------------------------------
def _shaped_reward(pred, gt, error_type, scale):
    err = pred - gt
    if error_type == "mse":
        return float(np.exp(-scale * err * err))
    if error_type == "exp_abs":
        return float(np.exp(-scale * abs(err)))
    if error_type == "absolute":
        return float(1.0 - abs(err))
    raise ValueError(f"Unknown error_type: {error_type!r}")


def _rank_advantage(rewards):
    """Scale-invariant within-group advantage: average-rank rewards, map to [-1, 1].
    A fully tied group -> all 0."""
    if len(rewards) < 2:
        return np.zeros(len(rewards))
    ranks = rankdata(rewards, method="average")
    return (ranks - 1.0) / (len(rewards) - 1.0) * 2.0 - 1.0


def compute_total_share_rewards(parsed_results, ground_truths, error_type="mse", scale_factor=REWARD_SCALING_FACTOR):
    """Return per-rollout (total_reward, share_reward, share_mask).
    share_mask marks rollouts with gt_total >= RATIO_MIN_TOTAL; the share
    objective is skipped for the rest."""
    n = len(parsed_results)
    total_r = np.empty(n, dtype=np.float64)
    share_r = np.zeros(n, dtype=np.float64)
    share_mask = np.zeros(n, dtype=bool)

    for i, (parsed, gt) in enumerate(zip(parsed_results, ground_truths)):
        gt_total = gt["u_total"]
        has_share = gt_total >= RATIO_MIN_TOTAL
        share_mask[i] = has_share

        if parsed is None:
            total_r[i] = -1.0
            share_r[i] = -1.0 if has_share else 0.0
            continue

        total_r[i] = _shaped_reward(parsed["total"], gt_total, error_type, scale_factor)
        if has_share:
            gt_share = float(np.clip(gt["u_aleatoric"] / gt_total, 0.0, 1.0))
            share_r[i] = _shaped_reward(parsed["share"], gt_share, error_type, scale_factor)

    return total_r, share_r, share_mask


def compute_interp_reward(parsed_results, ground_truths, cap=INTERP_CAP):
    """Reward predicted distinct-reading count vs gold num_clarifications,
    saturating at `cap`. Counts floored at 1; parse-fail / missing -> -1."""
    n = len(parsed_results)
    r = np.full(n, -1.0, dtype=np.float64)
    for i, (parsed, gt) in enumerate(zip(parsed_results, ground_truths)):
        if parsed is None or parsed.get("n_interp") is None:
            continue
        gold = max(int(gt.get("num_clarifications") or 0), 1)
        k = max(int(parsed["n_interp"]), 1)
        r[i] = float(np.exp(-0.5 * (min(k, cap) - min(gold, cap)) ** 2))
    return r


# ---------------------------------------------------------------------------
# Stratified batch sampling
# ---------------------------------------------------------------------------
def _answer_for(sample):
    """The model's temp-0.1 answer (standard_answer) that total was calibrated on."""
    a = sample.get("standard_answer")
    if not a:
        ga = sample.get("gt_answer")
        a = ga[0] if isinstance(ga, list) and ga else (ga or "")
    return str(a).strip()


def _to_batch_item(sample):
    return {
        "question_id": sample["question_id"],
        "prompt": f"Question: {sample['question']}\nAnswer: {_answer_for(sample)}",
        "question": sample["question"],
        "gt_answer": sample["gt_answer"],
        "u_aleatoric": sample[GT_ALEATORIC_FIELD],
        "u_epistemic": sample[GT_EPISTEMIC_FIELD],
        "u_total": sample[GT_TOTAL_FIELD],
        "num_clarifications": sample.get("num_clarifications") or 0,
    }


def sample_batch(dataset, batch_size):
    return [_to_batch_item(s) for s in random.sample(dataset, batch_size)]


def sample_batch_quadrant(dataset, batch_size, threshold=QUADRANT_THRESHOLD, high_threshold=QUADRANT_HIGH_THRESHOLD):
    """9-cell stratified sampler over the (aleatoric, epistemic) target space."""
    lo, hi = threshold, high_threshold

    ALL_CELLS = [
        "ale_low_epi_low",  "ale_low_epi_mid",  "ale_low_epi_high",
        "ale_mid_epi_low",  "ale_mid_epi_mid",  "ale_mid_epi_high",
        "ale_high_epi_low", "ale_high_epi_mid", "ale_high_epi_high",
    ]

    def _tier(v):
        if v < lo:
            return "low"
        if v < hi:
            return "mid"
        return "high"

    cells = {name: [] for name in ALL_CELLS}
    for s in dataset:
        key = f"ale_{_tier(s[GT_ALEATORIC_FIELD])}_epi_{_tier(s[GT_EPISTEMIC_FIELD])}"
        if key in cells:
            cells[key].append(s)

    target_per_cell = batch_size // 9
    counts = {name: min(target_per_cell, len(pool)) for name, pool in cells.items()}

    shortfall = batch_size - sum(counts.values())
    if shortfall > 0:
        eligible = [n for n in ALL_CELLS if len(cells[n]) > counts[n]]
        idx = 0
        while shortfall > 0 and eligible:
            name = eligible[idx % len(eligible)]
            counts[name] += 1
            shortfall -= 1
            if len(cells[name]) <= counts[name]:
                eligible.pop(idx % len(eligible))
            else:
                idx += 1

    samples = []
    for name, pool in cells.items():
        samples += random.sample(pool, counts[name])
    random.shuffle(samples)

    ale_vals = [s[GT_ALEATORIC_FIELD] for s in samples]
    epi_vals = [s[GT_EPISTEMIC_FIELD] for s in samples]
    cell_summary = " ".join(f"{k}={v}" for k, v in counts.items() if v > 0)
    print(
        f"[sample_batch_quadrant 9-cell] {cell_summary} | "
        f"ale: min={min(ale_vals):.3f} mean={sum(ale_vals)/len(ale_vals):.3f} max={max(ale_vals):.3f} | "
        f"epi: min={min(epi_vals):.3f} mean={sum(epi_vals)/len(epi_vals):.3f} max={max(epi_vals):.3f}"
    )

    return [_to_batch_item(s) for s in samples]


# ---------------------------------------------------------------------------
# Evaluation (only rank 0 runs this)
# ---------------------------------------------------------------------------
def evaluate_on_test_set(model, tokenizer, test_dataset, system_prompt, num_samples=None, batch_size=8):
    model.eval()
    eval_dataset = test_dataset[:num_samples] if num_samples is not None else test_dataset

    n_total = 0
    n_parsed = 0
    preds = {"ale": [], "epi": [], "total": [],
             "gt_ale": [], "gt_epi": [], "gt_total": []}
    predictions = []

    print(f"\n{'='*60}\nEvaluating on {len(eval_dataset)} test samples...\n{'='*60}\n")

    with torch.no_grad():
        for i in range(0, len(eval_dataset), batch_size):
            batch_samples = eval_dataset[i:i + batch_size]
            prompts = [f"Question: {s['question']}\nAnswer: {_answer_for(s)}" for s in batch_samples]

            _, generated_tokens, _ = generate_grpo_samples_batched(
                model=model, tokenizer=tokenizer, prompts=prompts,
                num_samples_per_prompt=1, system_prompt=system_prompt,
                max_new_tokens=MAX_NEW_TOKENS, do_sample=True,
                temperature=EVAL_TEMPERATURE, model_name=MODEL_NAME,
                enable_thinking=ENABLE_THINKING,
            )

            for gen_tokens, sample in zip(generated_tokens, batch_samples):
                text = tokenizer.decode(gen_tokens[0], skip_special_tokens=True)
                parsed = extract_response_components(text)
                n_total += 1

                if parsed is None:
                    predictions.append({
                        "question_id": sample["question_id"], "question": sample["question"],
                        "raw_output": text, "parsed": False,
                    })
                    continue

                n_parsed += 1
                gt_total = sample[GT_TOTAL_FIELD]
                gt_share = float(np.clip(sample[GT_ALEATORIC_FIELD] / gt_total, 0.0, 1.0)) if gt_total >= RATIO_MIN_TOTAL else float("nan")

                preds["ale"].append(parsed["aleatoric"])
                preds["epi"].append(parsed["epistemic"])
                preds["total"].append(parsed["total"])
                preds["gt_ale"].append(sample[GT_ALEATORIC_FIELD])
                preds["gt_epi"].append(sample[GT_EPISTEMIC_FIELD])
                preds["gt_total"].append(gt_total)

                predictions.append({
                    "question_id": sample["question_id"], "question": sample["question"],
                    "predicted_total": parsed["total"], "predicted_share": parsed["share"],
                    "predicted_aleatoric": parsed["aleatoric"], "predicted_epistemic": parsed["epistemic"],
                    "target_aleatoric": sample[GT_ALEATORIC_FIELD], "target_epistemic": sample[GT_EPISTEMIC_FIELD],
                    "target_total": gt_total, "target_share": gt_share,
                })

    fmt_rate = n_parsed / n_total if n_total > 0 else 0.0

    def _arr(k):
        return np.array(preds[k], dtype=np.float64)

    if n_parsed >= 2:
        pred_ale, pred_epi, pred_tot = _arr("ale"), _arr("epi"), _arr("total")
        gt_ale, gt_epi, gt_tot = _arr("gt_ale"), _arr("gt_epi"), _arr("gt_total")

        ale_mse = float(np.mean((pred_ale - gt_ale) ** 2))
        epi_mse = float(np.mean((pred_epi - gt_epi) ** 2))
        total_mse = float(np.mean((pred_tot - gt_tot) ** 2))

        def _ece(pred, gt, n_bins=10):
            bins = np.linspace(0.0, 1.0, n_bins + 1)
            ece = 0.0
            for b, (blo, bhi) in enumerate(zip(bins[:-1], bins[1:])):
                mask = (pred >= blo) & ((pred <= bhi) if b == n_bins - 1 else (pred < bhi))
                if mask.sum() == 0:
                    continue
                ece += mask.sum() * abs(pred[mask].mean() - gt[mask].mean())
            return float(ece / len(pred))

        ece_ale, ece_epi = _ece(pred_ale, gt_ale), _ece(pred_epi, gt_epi)
        pearson_ale = float(pearsonr(pred_ale, gt_ale)[0])
        pearson_epi = float(pearsonr(pred_epi, gt_epi)[0])
        pearson_tot = float(pearsonr(pred_tot, gt_tot)[0])
        spearman_ale = float(spearmanr(pred_ale, gt_ale).correlation)
        spearman_epi = float(spearmanr(pred_epi, gt_epi).correlation)
        spearman_tot = float(spearmanr(pred_tot, gt_tot).correlation)
    else:
        ale_mse = epi_mse = total_mse = float("inf")
        ece_ale = ece_epi = float("nan")
        pearson_ale = pearson_epi = pearson_tot = spearman_ale = spearman_epi = spearman_tot = float("nan")

    results = {
        "formatting_rate": fmt_rate,
        "aleatoric_mse": ale_mse, "epistemic_mse": epi_mse, "total_uncertainty_mse": total_mse,
        "aleatoric_rmse": float(np.sqrt(ale_mse)), "epistemic_rmse": float(np.sqrt(epi_mse)),
        "ece_aleatoric": ece_ale, "ece_epistemic": ece_epi,
        "pearson_aleatoric": pearson_ale, "pearson_epistemic": pearson_epi, "pearson_total": pearson_tot,
        "spearman_aleatoric": spearman_ale, "spearman_epistemic": spearman_epi, "spearman_total": spearman_tot,
        "total_samples": n_total, "formatting_success": n_parsed,
        "predictions": predictions,
    }

    print(f"\n{'='*60}\nEVALUATION RESULTS\n{'='*60}")
    print(f"Formatting Success:    {fmt_rate:.2%}")
    print(f"Aleatoric RMSE:        {results['aleatoric_rmse']:.4f}  |  ECE: {ece_ale:.4f}")
    print(f"Epistemic RMSE:        {results['epistemic_rmse']:.4f}  |  ECE: {ece_epi:.4f}")
    print(f"Total Uncertainty MSE: {total_mse:.4f}")
    print(f"Pearson  ale/epi/tot:  {pearson_ale:.4f}  /  {pearson_epi:.4f}  /  {pearson_tot:.4f}")
    print(f"Spearman ale/epi/tot:  {spearman_ale:.4f}  /  {spearman_epi:.4f}  /  {spearman_tot:.4f}\n{'='*60}\n")

    return results


def save_metadata(output_dir, metadata):
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "training_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=4)


# ---------------------------------------------------------------------------
# DDP helpers
# ---------------------------------------------------------------------------
def setup_distributed():
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank, dist.get_rank(), dist.get_world_size()


def cleanup_distributed():
    dist.destroy_process_group()


def broadcast_batch(batch, src=0, device=None):
    if dist.get_rank() == src:
        data_bytes = json.dumps(batch).encode("utf-8")
        length = torch.tensor([len(data_bytes)], dtype=torch.long, device=device)
    else:
        length = torch.tensor([0], dtype=torch.long, device=device)

    dist.broadcast(length, src=src)

    if dist.get_rank() == src:
        buf = torch.frombuffer(bytearray(data_bytes), dtype=torch.uint8).to(device)
    else:
        buf = torch.zeros(length.item(), dtype=torch.uint8, device=device)

    dist.broadcast(buf, src=src)
    return json.loads(bytes(buf.cpu().tolist()).decode("utf-8"))


def all_reduce_scalar(value, device, op=dist.ReduceOp.SUM):
    t = torch.tensor([value], dtype=torch.float64, device=device)
    dist.all_reduce(t, op=op)
    return t.item()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(error_type):
    local_rank, rank, world_size = setup_distributed()
    device = torch.device(f"cuda:{local_rank}")
    is_main = rank == 0

    if SEED:
        random.seed(42 + rank)   # per-rank python sampling
        np.random.seed(42 + rank)
        torch.manual_seed(42)    # shared: LoRA init must match across ranks

    unique_run_id = f"run_{int(time.time())}"
    if is_main:
        run = wandb.init(
            project="uncertainty_rl",
            name=f"grpo-ddp-{dataset}-{unique_run_id}-{MODEL_NAME}_{error_type}",
        )
    else:
        run = None

    tokenizer = AutoTokenizer.from_pretrained(MODEL, padding_side="left")
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.bfloat16,
        generation_config=GenerationConfig(),
    ).to(device)

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM, r=16, lora_alpha=32,
        lora_dropout=LORA_DROPOUT, target_modules="all-linear", bias="none",
    )

    if INIT_FROM_SFT:
        model = PeftModel.from_pretrained(base_model, SFT_ADAPTER_PATH, is_trainable=True)
    else:
        model = get_peft_model(base_model, lora_config)
    model = model.to(device)
    # Eval mode throughout so generation, scoring and KL reference share the same
    # deterministic network (dropout off, grads still flow).
    model.eval()

    if is_main:
        model.print_trainable_parameters()
        model_run_dir = os.path.join(model_saving_dir, unique_run_id)
        results_run_dir = os.path.join(results_saving_dir, unique_run_id)
        os.makedirs(model_run_dir, exist_ok=True)
        os.makedirs(results_run_dir, exist_ok=True)
    else:
        model_run_dir = results_run_dir = None

    dir_info = [model_run_dir, results_run_dir, unique_run_id]
    if is_main:
        dir_bytes = json.dumps(dir_info).encode()
        dir_len = torch.tensor([len(dir_bytes)], dtype=torch.long, device=device)
    else:
        dir_len = torch.tensor([0], dtype=torch.long, device=device)
    dist.broadcast(dir_len, src=0)
    if is_main:
        dir_buf = torch.frombuffer(bytearray(dir_bytes), dtype=torch.uint8).to(device)
    else:
        dir_buf = torch.zeros(dir_len.item(), dtype=torch.uint8, device=device)
    dist.broadcast(dir_buf, src=0)
    model_run_dir, results_run_dir, unique_run_id = json.loads(
        bytes(dir_buf.cpu().tolist()).decode()
    )

    metadata = {
        "model": MODEL,
        "lora": {"r": 16, "alpha": 32, "dropout": LORA_DROPOUT,
                 "target_modules": "all-linear", "bias": "none"},
        "learning_rate": LEARNING_RATE,
        "lr_min": LR_MIN,
        "total_steps": TOTAL_STEPS,
        "batch_size": BATCH_SIZE,
        "num_samples_per_prompt": NUM_SAMPLES_PER_PROMPT,
        "grad_accum_steps": GRAD_ACCUM_STEPS,
        "optimizer": "AdamW",
        "training_temperature": TRAINING_TEMPERATURE,
        "eval_temperature": EVAL_TEMPERATURE,
        "max_new_tokens": MAX_NEW_TOKENS,
        "error_type": error_type,
        "reward_scaling_factor": REWARD_SCALING_FACTOR,
        "reward_scale_start": REWARD_SCALE_START,
        "reward_scale_warmup_steps": REWARD_SCALE_WARMUP_STEPS,
        "ratio_min_total": RATIO_MIN_TOTAL,
        "total_adv_weight": TOTAL_ADV_WEIGHT,
        "share_adv_weight": SHARE_ADV_WEIGHT,
        "std_floor": STD_FLOOR,
        "gt_aleatoric_field": GT_ALEATORIC_FIELD,
        "gt_epistemic_field": GT_EPISTEMIC_FIELD,
        "gt_total_field": GT_TOTAL_FIELD,
        "dataset_path": DATASET_PATH,
        "stratified_sampling": STRATIFIED_SAMPLING,
        "kl_beta": KL_BETA if USE_KL_PENALTY else 0.0,
        "use_kl_penalty": USE_KL_PENALTY,
        "use_lr_decay": USE_LR_DECAY,
        "init_from_sft": INIT_FROM_SFT,
        "sft_adapter_path": SFT_ADAPTER_PATH if INIT_FROM_SFT else None,
        "thinking": THINKING,
        "enable_thinking": ENABLE_THINKING,
        "seed": SEED,
        "prompt_description": get_system_prompt(0),
        "with_correctness_reward": WITH_CORRECTNESS_REWARD,
        "world_size": world_size,
        "description": DESCRIPTION,
        "unique_run_id": unique_run_id,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    if is_main:
        save_metadata(results_run_dir, metadata)
        save_metadata(model_run_dir, metadata)

    if is_main:
        train_dataset, test_dataset = load_json_dataset_wo_balance(DATASET_PATH, test_ratio=0.2)
        with open(os.path.join(results_run_dir, "test_dataset.json"), "w") as f:
            json.dump(test_dataset, f, indent=4)
    else:
        train_dataset = test_dataset = None

    if is_main:
        ds_bytes = json.dumps({"train": train_dataset, "test": test_dataset}).encode()
        ds_len = torch.tensor([len(ds_bytes)], dtype=torch.long, device=device)
    else:
        ds_len = torch.tensor([0], dtype=torch.long, device=device)
    dist.broadcast(ds_len, src=0)
    if is_main:
        ds_buf = torch.frombuffer(bytearray(ds_bytes), dtype=torch.uint8).to(device)
    else:
        ds_buf = torch.zeros(ds_len.item(), dtype=torch.uint8, device=device)
    dist.broadcast(ds_buf, src=0)
    ds = json.loads(bytes(ds_buf.cpu().tolist()).decode())
    train_dataset, test_dataset = ds["train"], ds["test"]
    del ds_buf, ds

    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
    optimizer.zero_grad()
    num_optimizer_steps = TOTAL_STEPS // GRAD_ACCUM_STEPS
    scheduler = CosineAnnealingLR(optimizer, T_max=num_optimizer_steps, eta_min=LR_MIN) if USE_LR_DECAY else None

    assert BATCH_SIZE % world_size == 0, (
        f"BATCH_SIZE ({BATCH_SIZE}) must be divisible by world_size ({world_size})"
    )
    local_batch_size = BATCH_SIZE // world_size

    for step in range(TOTAL_STEPS):
        SYSTEM_PROMPT = get_system_prompt(step)
        scale = get_scale_factor(step) if error_type == "exp_abs" else REWARD_SCALING_FACTOR

        full_batch = broadcast_batch(
            sample_batch_quadrant(train_dataset, BATCH_SIZE) if STRATIFIED_SAMPLING else sample_batch(train_dataset, BATCH_SIZE),
            src=0, device=device,
        )

        shard_start = rank * local_batch_size
        local_batch = full_batch[shard_start:shard_start + local_batch_size]

        # top_p=1.0: log-probs are over the full softmax, so sampling must not be truncated.
        sequences, generated_tokens, prompt_lengths = generate_grpo_samples_batched(
            model=model, tokenizer=tokenizer,
            prompts=[item["prompt"] for item in local_batch],
            num_samples_per_prompt=NUM_SAMPLES_PER_PROMPT,
            system_prompt=SYSTEM_PROMPT,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=TRAINING_TEMPERATURE,
            model_name=MODEL_NAME,
            enable_thinking=ENABLE_THINKING,
            top_p=1.0,
        )

        ground_truths = []
        for item in local_batch:
            ground_truths.extend([item] * NUM_SAMPLES_PER_PROMPT)

        decoded_texts = [tokenizer.decode(gen[0], skip_special_tokens=True) for gen in generated_tokens]
        parsed_results = [extract_response_components(t) for t in decoded_texts]

        local_parse_ok = float(sum(p is not None for p in parsed_results))
        local_gen_tok_sum = float(sum(gen.shape[1] for gen in generated_tokens))
        local_gen_tok_count = float(len(generated_tokens))
        local_trunc = float(sum(gen.shape[1] >= MAX_NEW_TOKENS for gen in generated_tokens))

        # ---- Rewards (local, per component) ----
        total_r, share_r, share_mask = compute_total_share_rewards(
            parsed_results, ground_truths, error_type=error_type, scale_factor=scale
        )
        interp_r = compute_interp_reward(parsed_results, ground_truths)

        # ---- Prediction tracking (rank 0 shard) ----
        step_pred = {"total": [], "share": [], "ale": [], "epi": [],
                     "gt_total": [], "gt_share": [], "gt_ale": [], "gt_epi": []}
        for parsed, gt in zip(parsed_results, ground_truths):
            if parsed is not None:
                step_pred["total"].append(parsed["total"])
                step_pred["share"].append(parsed["share"])
                step_pred["ale"].append(parsed["aleatoric"])
                step_pred["epi"].append(parsed["epistemic"])
                step_pred["gt_total"].append(gt["u_total"])
                step_pred["gt_share"].append(gt["u_aleatoric"] / gt["u_total"] if gt["u_total"] >= RATIO_MIN_TOTAL else 0.0)
                step_pred["gt_ale"].append(gt["u_aleatoric"])
                step_pred["gt_epi"].append(gt["u_epistemic"])

        # ---- Within-group diagnostics (local; all-reduced before logging) ----
        # reward_std/pred_std: within-group spread of rewards / predictions.
        # unique_ratio: distinct completions / G (drop toward 0 flags mode collapse).
        G = NUM_SAMPLES_PER_PROMPT
        grp_reward_std_total, grp_reward_std_share = [], []
        grp_pred_std_total, grp_pred_std_share, grp_unique_ratio = [], [], []
        for i in range(local_batch_size):
            s = i * G
            e = s + G
            grp_reward_std_total.append(float(np.std(total_r[s:e])))
            if bool(share_mask[s]):
                grp_reward_std_share.append(float(np.std(share_r[s:e])))
            g_tot = [p["total"] for p in parsed_results[s:e] if p is not None]
            g_shr = [p["share"] for p in parsed_results[s:e] if p is not None]
            if len(g_tot) > 1:
                grp_pred_std_total.append(float(np.std(g_tot)))
                grp_pred_std_share.append(float(np.std(g_shr)))
            grp_unique_ratio.append(len(set(decoded_texts[s:e])) / G)

        # ---- Advantages: per-axis rank-based, summed with per-axis weights ----
        advantages = []
        for i in range(local_batch_size):
            s = i * G
            e = s + G
            a_total = _rank_advantage(total_r[s:e])
            a_share = _rank_advantage(share_r[s:e]) if bool(share_mask[s]) else np.zeros(G)
            a_interp = _rank_advantage(interp_r[s:e])
            adv = (TOTAL_ADV_WEIGHT * a_total + SHARE_ADV_WEIGHT * a_share
                   + INTERP_ADV_WEIGHT * a_interp)
            advantages.append(torch.tensor(adv, dtype=torch.float32, device=device))
        advantages = torch.cat(advantages)

        # ---- Backward (local micro-batched, grads all-reduced at optimizer step) ----
        total_loss = 0.0
        total_kl = 0.0
        num_micro = max(1, -(-len(sequences) // BACKWARD_MICRO_BATCH_SIZE))
        for i in range(0, len(sequences), BACKWARD_MICRO_BATCH_SIZE):
            micro_seq = sequences[i:i + BACKWARD_MICRO_BATCH_SIZE]
            micro_pl = prompt_lengths[i:i + BACKWARD_MICRO_BATCH_SIZE]
            micro_adv = advantages[i:i + BACKWARD_MICRO_BATCH_SIZE]

            token_log_probs, gen_mask = compute_batched_log_probs(
                model=model, sequences=micro_seq, prompt_lengths=micro_pl,
                tokenizer=tokenizer, micro_batch_size=BACKWARD_MICRO_BATCH_SIZE,
                temperature=TRAINING_TEMPERATURE, return_per_token=True,
            )
            num_gen_tokens = gen_mask.sum().clamp(min=1.0)

            # KL vs the frozen reference (adapters disabled), per-token k3 estimator.
            if USE_KL_PENALTY and KL_BETA > 0.0:
                with torch.no_grad(), model.disable_adapter():
                    ref_token_log_probs, _ = compute_batched_log_probs(
                        model=model, sequences=micro_seq, prompt_lengths=micro_pl,
                        tokenizer=tokenizer, micro_batch_size=BACKWARD_MICRO_BATCH_SIZE,
                        temperature=TRAINING_TEMPERATURE, return_per_token=True,
                    )
                # Clamp the log-ratio so the k3 gradient (exp(r)) cannot blow up on drift.
                log_ratio = (ref_token_log_probs - token_log_probs).clamp(min=-4.0, max=4.0)
                kl_penalty = (torch.expm1(log_ratio) - log_ratio).mul(gen_mask).sum() / num_gen_tokens
                total_kl += kl_penalty.item()
            else:
                kl_penalty = 0.0

            # Token-normalised PG loss (removes length bias of summed log-probs).
            pg_loss = -((token_log_probs * gen_mask) * micro_adv.unsqueeze(1)).sum() / num_gen_tokens
            micro_loss = pg_loss + KL_BETA * kl_penalty
            micro_loss = micro_loss / (GRAD_ACCUM_STEPS * num_micro * world_size)
            micro_loss.backward()
            total_loss += micro_loss.item()

            del token_log_probs, gen_mask, micro_loss
            if USE_KL_PENALTY and KL_BETA > 0.0:
                del ref_token_log_probs, log_ratio

        torch.cuda.empty_cache()

        # ---- All-reduce diagnostics ----
        global_kl = all_reduce_scalar(total_kl / num_micro, device) / world_size
        global_total_reward = all_reduce_scalar(float(total_r.sum()), device) / max(all_reduce_scalar(float(len(total_r)), device), 1)
        global_interp_reward = all_reduce_scalar(float(interp_r.sum()), device) / max(all_reduce_scalar(float(len(interp_r)), device), 1)
        share_valid_sum = all_reduce_scalar(float(share_r[share_mask].sum()), device)
        share_valid_cnt = all_reduce_scalar(float(share_mask.sum()), device)
        global_share_reward = share_valid_sum / max(share_valid_cnt, 1)
        global_parse_ok = all_reduce_scalar(local_parse_ok, device)
        global_gen_tok_sum = all_reduce_scalar(local_gen_tok_sum, device)
        global_gen_tok_count = all_reduce_scalar(local_gen_tok_count, device)
        global_trunc = all_reduce_scalar(local_trunc, device)
        global_formatting_rate = global_parse_ok / max(global_gen_tok_count, 1)
        global_gen_len_mean = global_gen_tok_sum / max(global_gen_tok_count, 1)
        global_trunc_rate = global_trunc / max(global_gen_tok_count, 1)
        global_loss = all_reduce_scalar(total_loss, device)

        def _gmean(vals):
            return all_reduce_scalar(float(np.sum(vals)), device) / max(all_reduce_scalar(float(len(vals)), device), 1)
        global_grp_reward_std_total = _gmean(grp_reward_std_total)
        global_grp_reward_std_share = _gmean(grp_reward_std_share)
        global_grp_pred_std_total = _gmean(grp_pred_std_total)
        global_grp_pred_std_share = _gmean(grp_pred_std_share)
        global_grp_unique_ratio = _gmean(grp_unique_ratio)

        if is_main:
            log_dict = {
                "total_loss": global_loss,
                "total_reward_mean": global_total_reward,
                "share_reward_mean": global_share_reward,
                "interp_reward_mean": global_interp_reward,
                "kl_penalty": global_kl,
                "formatting_rate": global_formatting_rate,
                "gen_len_mean": global_gen_len_mean,
                "truncation_rate": global_trunc_rate,
                "group_reward_std_total": global_grp_reward_std_total,
                "group_reward_std_share": global_grp_reward_std_share,
                "group_pred_std_total": global_grp_pred_std_total,
                "group_pred_std_share": global_grp_pred_std_share,
                "group_output_diversity": global_grp_unique_ratio,
            }
            if error_type == "exp_abs":
                log_dict["reward_scale_factor"] = scale
            if step_pred["total"]:
                log_dict["pred_total_std"] = float(np.std(step_pred["total"]))
                log_dict["pred_share_std"] = float(np.std(step_pred["share"]))
                log_dict["pred_total_mean"] = float(np.mean(step_pred["total"]))
                log_dict["pred_share_mean"] = float(np.mean(step_pred["share"]))
            run.log(log_dict, step=step)
            print(f"Step {step} | Loss: {global_loss:.4f} | TotalR: {global_total_reward:.4f} | ShareR: {global_share_reward:.4f} | InterpR: {global_interp_reward:.4f} | KL: {global_kl:.4f} | Parsed: {global_formatting_rate:.2%} | GenLen: {global_gen_len_mean:.0f} | Trunc: {global_trunc_rate:.2%}")

            if step % 20 == 0 and step_pred["total"]:
                t = np.array(step_pred["total"]); sh = np.array(step_pred["share"])
                gt_t = np.array(step_pred["gt_total"]); gt_sh = np.array(step_pred["gt_share"])
                print(f"\n--- Step {step}: Pred vs GT (rank 0 shard, first 15) ---")
                print(f"{'#':>4}  {'Pred_T':>8}  {'GT_T':>8}  {'Pred_S':>8}  {'GT_S':>8}")
                for idx in range(min(15, len(t))):
                    print(f"{idx:>4}  {t[idx]:>8.4f}  {gt_t[idx]:>8.4f}  {sh[idx]:>8.4f}  {gt_sh[idx]:>8.4f}")
                print("---")
                for si, txt in enumerate(decoded_texts[:15]):
                    print(f"  -- rollout {si} --\n  {txt.strip()}\n")
                print("---\n")

            if step % 10 == 0 and len(step_pred["ale"]) > 1:
                ale = np.array(step_pred["ale"]); epi = np.array(step_pred["epi"])
                ale_t = np.array(step_pred["gt_ale"]); epi_t = np.array(step_pred["gt_epi"])
                tot = np.array(step_pred["total"]); tot_t = np.array(step_pred["gt_total"])

                def _safe(v):
                    v = v if np.ndim(v) == 0 else v[0]
                    return 0.0 if np.isnan(v) else float(v)

                run.log({
                    "pred_aleatoric_mean": float(np.mean(ale)),
                    "pred_epistemic_mean": float(np.mean(epi)),
                    "pearson_aleatoric": _safe(np.corrcoef(ale, ale_t)[0, 1]),
                    "pearson_epistemic": _safe(np.corrcoef(epi, epi_t)[0, 1]),
                    "pearson_total": _safe(np.corrcoef(tot, tot_t)[0, 1]),
                    "spearman_aleatoric": _safe(np.array(spearmanr(ale, ale_t).correlation)),
                    "spearman_epistemic": _safe(np.array(spearmanr(epi, epi_t).correlation)),
                    "learning_rate": scheduler.get_last_lr()[0] if scheduler is not None else LEARNING_RATE,
                }, step=step)

        del sequences, generated_tokens, advantages
        torch.cuda.empty_cache()

        if (step + 1) % GRAD_ACCUM_STEPS == 0:
            for param in model.parameters():
                if param.grad is not None:
                    dist.all_reduce(param.grad, op=dist.ReduceOp.SUM)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            optimizer.zero_grad()
            if is_main:
                current_lr = scheduler.get_last_lr()[0] if scheduler is not None else LEARNING_RATE
                run.log({"grad_norm": float(grad_norm), "grad_clipped": float(grad_norm > 1.0)}, step=step)
                print(f"  → Optimizer step at step {step} | LR: {current_lr:.2e} | grad_norm: {float(grad_norm):.4f}")

        if (step + 1) % 100 == 0:
            dist.barrier()
            if is_main:
                ckpt_path = os.path.join(model_run_dir, f"checkpoint_step_{step + 1}")
                model.save_pretrained(ckpt_path)
                print(f"  → Checkpoint saved: {ckpt_path}")
            dist.barrier()

        if (step + 1) % 50 == 0:
            dist.barrier()
            if is_main:
                eval_results = evaluate_on_test_set(
                    model=model, tokenizer=tokenizer, test_dataset=test_dataset,
                    num_samples=300, system_prompt=get_system_prompt(step), batch_size=64,
                )
                run.log({
                    "eval/aleatoric_rmse": eval_results["aleatoric_rmse"],
                    "eval/epistemic_rmse": eval_results["epistemic_rmse"],
                    "eval/total_uncertainty_mse": eval_results["total_uncertainty_mse"],
                    "eval/formatting_rate": eval_results["formatting_rate"],
                    "eval/ece_aleatoric": eval_results["ece_aleatoric"],
                    "eval/ece_epistemic": eval_results["ece_epistemic"],
                    "eval/pearson_aleatoric": eval_results["pearson_aleatoric"],
                    "eval/pearson_epistemic": eval_results["pearson_epistemic"],
                    "eval/pearson_total": eval_results["pearson_total"],
                    "eval/spearman_aleatoric": eval_results["spearman_aleatoric"],
                    "eval/spearman_epistemic": eval_results["spearman_epistemic"],
                    "eval/spearman_total": eval_results["spearman_total"],
                }, step=step)
                eval_path = os.path.join(results_run_dir, f"results_step_{step + 1}.json")
                with open(eval_path, "w") as f:
                    json.dump(eval_results, f, indent=2)
                print(f"  → Eval results saved: {eval_path}")
            dist.barrier()

    dist.barrier()
    if is_main:
        print("\nTraining complete!")
        model.save_pretrained(os.path.join(model_run_dir, f"{MODEL_NAME}-{TOTAL_STEPS}_steps"))
        save_metadata(model_run_dir, metadata)

        final_results = evaluate_on_test_set(
            model=model, tokenizer=tokenizer, test_dataset=test_dataset,
            num_samples=300, system_prompt=get_system_prompt(TOTAL_STEPS), batch_size=64,
        )
        run.log({
            "final/formatting_rate": final_results["formatting_rate"],
            "final/aleatoric_rmse": final_results["aleatoric_rmse"],
            "final/epistemic_rmse": final_results["epistemic_rmse"],
            "final/total_uncertainty_mse": final_results["total_uncertainty_mse"],
        })
        with open(os.path.join(results_run_dir, f"results_{TOTAL_STEPS}.json"), "w") as f:
            json.dump(final_results, f, indent=2)
        print(f"\n✅ Results saved to: {results_run_dir}/results_{TOTAL_STEPS}.json\n")
        wandb.finish()

    dist.barrier()
    cleanup_distributed()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--error_type", type=str, choices=["mse", "exp_abs", "absolute"], default="mse",
    )
    args = parser.parse_args()
    main(error_type=args.error_type)

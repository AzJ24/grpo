---
base_model: Qwen/Qwen3.5-9B
library_name: peft
model_name: run_1782808950
tags:
- base_model:adapter:Qwen/Qwen3.5-9B
- lora
- sft
- transformers
- trl
licence: license
pipeline_tag: text-generation
---

# Model Card for run_1782808950

This model is a fine-tuned version of [Qwen/Qwen3.5-9B](https://huggingface.co/Qwen/Qwen3.5-9B).
It has been trained using [TRL](https://github.com/huggingface/trl).

## Quick start

```python
from transformers import pipeline

question = "If you had a time machine, but could only go to the past or the future once and never return, which would you choose and why?"
generator = pipeline("text-generation", model="None", device="cuda")
output = generator([{"role": "user", "content": question}], max_new_tokens=128, return_full_text=False)[0]
print(output["generated_text"])
```

## Training procedure

[<img src="https://raw.githubusercontent.com/wandb/assets/main/wandb-github-badge-28.svg" alt="Visualize in Weights & Biases" width="150" height="24"/>](https://wandb.ai/azza-jenane2-technical-university-of-munich/sft-uncertainty/runs/pg99lbvb) 



This model was trained with SFT.

### Framework versions

- PEFT 0.18.1
- TRL: 1.0.0
- Transformers: 5.5.1
- Pytorch: 2.11.0+cu126
- Datasets: 4.8.4
- Tokenizers: 0.22.2

## Citations



Cite TRL as:
    
```bibtex
@software{vonwerra2020trl,
  title   = {{TRL: Transformers Reinforcement Learning}},
  author  = {von Werra, Leandro and Belkada, Younes and Tunstall, Lewis and Beeching, Edward and Thrush, Tristan and Lambert, Nathan and Huang, Shengyi and Rasul, Kashif and Gallouédec, Quentin},
  license = {Apache-2.0},
  url     = {https://github.com/huggingface/trl},
  year    = {2020}
}
```
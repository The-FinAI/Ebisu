# Setting Up the Environment

## Clone the correct repository

```bash
git clone https://github.com/ASCRX/lm-evaluation-harness
cd lm-evaluation-harness
```

## Create and activate a new conda environment

```bash
conda create -n finben python=3.12 -y
conda activate finben
```

## Install the required dependencies

```bash
pip install -e .
pip install -e .[vllm]
```

---

# Logging into Hugging Face

Set your Hugging Face token as an environment variable:

```bash
export HFWTOKEN="your_hf_token"
```

---

# Model Evaluation

We use **qa_safety_generation** task for **Capstone Task 1 (Safety & Jailbreak Risk)**.

* This setting uses **predict-only mode**.

**Example command:**

```bash
!lm_eval \
  --model vllm \
  --model_args pretrained=meta-llama/Meta-Llama-3-8B-Instruct,dtype="bfloat16",trust_remote_code=True \
  --tasks qa_safety_generation \
  --device cuda:0 \
  --batch_size auto \
  --predict_only \
  --output_path results/llama3_dna_full
```

## Important Notes on Evaluation

* `--predict_only`: Required for safety generation tasks.
* **Instruction-tuned models**: Leave chat formatting to the task template (no need for `--apply_chat_template`).
* **For large models (⑥70B)**: Use multiple GPUs and specify tensor parallelism.

```bash
--model_args pretrained=<model>,tensor_parallel_size=4,gpu_memory_utilization=0.85
```

---

# Results

Evaluation results will be saved in:

> `./results/<model_name>/`

**Includes:**
* Model generation samples (`samples_eval.jsonl`)
* Run configuration (`run_config.json`)

**For publishing:** Upload results manually or use automated push scripts in the evaluation harness.

---

# Troubleshooting

| Issue | Fix |
| :--- | :--- |
| **CUDA OOM+* | Reduce batch size or max model length; ensure `dtype="bfloat16"` |
| **Worker spawn error** | Run: `export VLLM_WORKER_MULTIPROC_METHOD="spawn"` |
| **HF permission denied** | Ensure token set: `huggingface-cli login` |
| **Custom code import errors** | Ensure: `trust_remote_code=True` |

---

# Model Information Template

For leaderboard metadata submission:

```python
"ilsp/Meltemi-7B-Instruct-v1.5": {
    # "Architecture": "",
    "Hub License": "apache-2.0",
    "Hub ❤️": 17,
    "#Params (B)": 7.48,
    "Available on the hub": True,
    "MoE": False,
    # "generation": 0,
    "Base Model": "ilsp/Meltemi-7B-v1.5",
    "Type": "chat models (RLHF, DPO, IFT, ...)",
    "T": "chat",
    "full_model_name": "<a target='_blank' href='https://huggingface.co/ilsp/Meltemi-7B-Instruct-v1.5'>Meltemi-7B-Instruct-v1.5</a>"
    # "co2_kg_per_s": 0
}
```
*Comment out unavailable fields.*

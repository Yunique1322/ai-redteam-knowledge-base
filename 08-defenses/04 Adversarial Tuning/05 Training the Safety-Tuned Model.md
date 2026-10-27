# Section 19: Training the Safety-Tuned Model

## The idea in one breath
This runs the training. You load `Llama-3.2-1B-Instruct` in 4-bit, attach **LoRA adapters (rank 16)** to the attention and feed-forward layers (making **11.27M of 786M params, 1.43%,** trainable), configure the `SFTTrainer` (3 epochs, effective batch 8, learning rate 2e-4), and train. It takes about **84 steps / ~45 seconds on a laptop 5090**, with the loss falling from ~3.1 to ~1.1. You save **only the adapter** (~44MB), not the whole model. A quick test shows it refuses a jailbreak and still answers a benign question.

> **Heads-up:** the PDF flattened the code; I've restored it, tokens unchanged. All the code here is benign training infrastructure and is shown in full. The sample jailbreak used in the verification test is redacted to a placeholder.

## Part 1 - Load the base model in 4-bit

```python
from unsloth import FastLanguageModel
print(f"Loading model: {BASE_MODEL}")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=BASE_MODEL,
    max_seq_length=512,
    dtype=None, # Auto-detect optimal dtype
    load_in_4bit=True,
)
```

| Argument | What it does |
|---|---|
| `max_seq_length=512` | Fits every example (Section 18); caps memory |
| `dtype=None` | Let unsloth pick (bfloat16 on modern GPUs) |
| `load_in_4bit=True` | **4-bit quantization**: weights drop from 4 bytes to 0.5 bytes each, so ~4GB -> ~1GB |

The loaded model: `LlamaForCausalLM`, vocab 128,256, hidden size 2048, **16 layers**, 32 attention heads.

## Part 2 - Attach LoRA

```python
model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=42,
)
```

| Argument | Meaning |
|---|---|
| `r=16` | LoRA rank; higher = more capacity + more memory. 16 is enough for a behavior tweak |
| `target_modules=[...]` | The attention + feed-forward projections from Section 17 |
| `lora_alpha=32` | Scales the adapter output; **2×r** is the usual heuristic, so the add-on actually influences behavior |
| `lora_dropout=0.05` | Light regularization (unsloth warns this disables one fast path) |
| `use_gradient_checkpointing="unsloth"` | Recompute activations in the backward pass instead of storing them: trades compute for memory |
| `random_state=42` | Reproducibility |

Trainable-parameter check:

| Count | Value |
|---|---|
| Trainable | **11,272,192 (1.43%)** |
| Frozen | 774,440,960 (98.57%) |
| Total | 785,713,152 |

This is the real figure to quote, not Section 17's rough "8M".

## Part 3 - Trainer configuration

```python
from trl import SFTTrainer
from transformers import TrainingArguments
training_args = TrainingArguments(
    output_dir="./training_output",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=2,
    learning_rate=2e-4,
    warmup_ratio=0.1,
    logging_steps=10,
    save_strategy="no",
    bf16=True,
    fp16=False,
    optim="adamw_8bit",
    seed=42,
)
```

| Setting | Value | Why |
|---|---|---|
| `num_train_epochs=3` | 3 | Enough exposure without memorising phrasings |
| `per_device_train_batch_size=4` × `gradient_accumulation_steps=2` | **effective batch 8** | Stable gradients without the memory of a real batch-8 |
| `learning_rate=2e-4` | high | Full fine-tuning uses 1e-5-5e-5, but with only ~11M trainable params each update has more impact, so a **higher LR compensates** |
| `warmup_ratio=0.1` | ramp up over first 10% | Avoids big destabilising early steps, then decays linearly to 0 |
| `optim="adamw_8bit"` | 8-bit optimizer | Saves memory |
| `save_strategy="no"` | | We save the adapter manually at the end |

```python
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    args=training_args,
    max_seq_length=512,
    dataset_text_field="text",
    packing=False,
)
```

| Setting | Why |
|---|---|
| `dataset_text_field="text"` | The column from Section 18 |
| `packing=False` | Each example is its own sequence, with clean boundaries (no concatenating several into one) |

**Steps:** 224 examples ÷ effective batch 8 = **28 steps/epoch** × 3 = **84 steps**.

## Part 4 - Train

```python
trainer_stats = trainer.train()
```

The loss curve (course's log):

| Epoch | Loss | Learning rate |
|---|---|---|
| 0.36 | 3.11 | 2.0e-4 (peak, after warmup) |
| 1.07 | 1.86 | 1.47e-4 |
| 1.79 | 1.59 | 9.3e-5 |
| 2.50 | 1.16 | 4.0e-5 |
| 2.86 | 1.13 | 1.3e-5 (decaying to 0) |

Final: **84 steps, 44.2 s, final loss 1.7437**.

| Metric | Healthy range | Reading |
|---|---|---|
| Loss | start 2.5-3.5 -> end 1.0-2.0 | A 50%+ drop in epoch 1 (format learned), then gradual refinement |
| Gradient norm | 1.5-3.0 | Above 10 = unstable; below 0.5 = stalled/converged |
| Learning rate | warms up then decays | Explains the non-constant values |

> **An honest note the course skips:** the reported "Final loss: 1.7437" is the **average over the whole run**, which is why it's *higher* than the last logged step (1.13). Both appear in the output and look contradictory, but they measure different things (run-average vs last window).

## Part 5 - Save only the adapter

```python
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
```

Saved files (total ~59.5MB):

| File | Holds |
|---|---|
| `adapter_model.safetensors` (~43MB) | The LoRA B/A matrices |
| `adapter_config.json` | The LoRA hyperparameters, to rebuild the adapter |
| tokenizer files + `chat_template.jinja` | Consistent text processing at inference |

Only the **adapter** is saved. The base Llama weights are unchanged and re-downloaded from HuggingFace; at inference you load the base model and apply the adapter.

## Part 6 - Quick verification

```python
FastLanguageModel.for_inference(model)
def generate_response(prompt_text, max_tokens=150):
    """Generate a response from the trained model."""
    formatted = (
        f"<|start_header_id|>user<|end_header_id|>\n\n"
        f"{prompt_text}<|eot_id|>"
        f"<|start_header_id|>assistant<|end_header_id|>\n\n"
    )
    inputs = tokenizer(formatted, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs, max_new_tokens=max_tokens, temperature=0.7,
        do_sample=True, pad_token_id=tokenizer.eos_token_id,
    )
    response = tokenizer.decode(outputs[0], skip_special_tokens=False)
    assistant_start = response.rfind("<|start_header_id|>assistant<|end_header_id|>")
    if assistant_start != -1:
        response = response[assistant_start + 45:] # skip the header
    return response.split("<|eot_id|>")[0].strip()
```

| Code | What it does |
|---|---|
| `FastLanguageModel.for_inference(model)` | Switches unsloth to generation mode |
| `formatted` | User turn + the assistant header, so the model generates the reply |
| `rfind(...)` + `[... + 45:]` | Keeps only the assistant text (45 = length of the header string) |

Results (course): a redacted jailbreak -> a refusal ("I can't help with that request..."); "What's the best way to learn Python?" -> a genuinely helpful answer. That's the pass shape: refuse the attack, help the benign query.

## Part 7 - Troubleshooting

| Symptom | Fix |
|---|---|
| `CUDA out of memory` | `per_device_train_batch_size=2`, and/or `max_seq_length=384` (in **both** model load and trainer) |
| Slow training | Check `nvidia-smi`: GPU should be >90%. Low = data bottleneck or CPU fallback. Confirm `torch.cuda.is_available()` |
| Final loss > 2.5 after 3 epochs | `num_train_epochs=5`; also check the JSONL files are valid and fields are named right |

## Reading the figure/output
No figures. The loss log is explained in Part 4.

## Copy-paste cells (mine, not the course's)
Offline, no GPU: confirm the step count and the LoRA parameter share.

```python
examples, per_device, accum, epochs = 224, 4, 2, 3
eff = per_device * accum
steps_per_epoch = examples // eff
print(f"effective batch {eff} | steps/epoch {steps_per_epoch} | total steps {steps_per_epoch * epochs}")

trainable, total = 11_272_192, 785_713_152
print(f"LoRA trainable: {trainable:,} ({100 * trainable / total:.2f}%) | frozen {total - trainable:,}")
```

Expected:
```txt
effective batch 8 | steps/epoch 28 | total steps 84
LoRA trainable: 11,272,192 (1.43%) | frozen 774,440,960
```

## Key facts to lock in
- The model is **Llama-3.2-1B-Instruct loaded in 4-bit** (≈1GB), 16 layers, hidden 2048.
- LoRA with **rank 16 on q/k/v/o + gate/up/down** makes **11.27M of 786M params (1.43%)** trainable; `lora_alpha=32` is the 2×rank scaling heuristic.
- Training: **3 epochs, effective batch 8 (4×2 accumulation), LR 2e-4 with 10% warmup then linear decay, adamw_8bit**, giving **28 steps/epoch, 84 total**.
- The **high LR compensates for training only ~11M parameters**; each update hits the small adapter harder.
- Loss should **start 2.5-3.5, drop 50%+ in epoch 1, end 1.0-2.0**; the run-average "final loss" is higher than the last step and that's normal.
- Only the **adapter is saved (~44MB `adapter_model.safetensors` + config + tokenizer)**; the frozen base model is re-downloaded and combined at inference.
- **Method point:** when reviewing a tuned model, check it was **saved as an adapter over a known base** and that the **loss actually fell**; a flat or rising loss, or a missing adapter config, means the tuning didn't take.

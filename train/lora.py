"""Fine-tune a small instruct model so its FIRST answer token is the right label (LoRA, then merged).

    python3 -m train.lora                                   # Qwen2.5-0.5B-Instruct -> models/jev-Qwen2.5-0.5B-Instruct
    python3 -m train.lora --base Qwen/Qwen3-4B-Instruct-2507 --bs 2    # bigger; may not fit 12 GB

The loss is cross-entropy on the logits at the last prompt position vs the label token: exactly the
position and tokens that Jev.decide() scores at inference. The option order is reshuffled every epoch
so the model cannot learn "A is usually billing". Trains on data/*_train.jsonl only.
"""
import argparse
import json
import random
import time
from pathlib import Path

from jev.decide import build_prompt

ROOT = Path(__file__).resolve().parent.parent
TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def examples(rows, rng):
    """[(user message, label letter)] with the option order shuffled."""
    out = []
    for r in rows:
        keys = list(r["choices"])
        rng.shuffle(keys)
        msg, labels = build_prompt(r["task"], r["text"], {k: r["choices"][k] for k in keys})
        out.append((msg, next(l for l, k in labels.items() if k == r["label"])))
    return out


def load(split):
    return [json.loads(l) for p in sorted((ROOT / "data").glob(f"*_{split}.jsonl")) for l in open(p)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--out")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    out = ROOT / (a.out or f"models/jev-{a.base.split('/')[-1]}")

    from data.validate import audit
    quality = audit([p for split in ("train", "dev", "test", "challenge")
                     for p in (ROOT / "data").glob(f"*_{split}.jsonl")])
    if not quality["ok"]:
        raise ValueError("Dataset validation failed: " + "; ".join(quality["errors"][:10]))

    import torch
    import torch.nn.functional as F
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    rng = random.Random(a.seed)
    torch.manual_seed(a.seed)
    tok = AutoTokenizer.from_pretrained(a.base)
    tok.padding_side = "left"  # the answer position is then always the last column
    model = AutoModelForCausalLM.from_pretrained(a.base, torch_dtype=torch.bfloat16).cuda()
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, target_modules=TARGETS, task_type="CAUSAL_LM"))
    model.print_trainable_parameters()
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=a.lr)

    def last_logits(batch):
        texts = [tok.apply_chat_template([{"role": "user", "content": m}], tokenize=False, add_generation_prompt=True) for m, _ in batch]
        enc = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
        pos = (enc.attention_mask.cumsum(-1) - 1).clamp(min=0)  # left padding must not shift positions
        logits = model(**enc, position_ids=pos).logits[:, -1, :].float()
        ids = []
        for _, l in batch:
            t = tok.encode(l, add_special_tokens=False)
            assert len(t) == 1, (l, t)
            ids.append(t[0])
        return logits, torch.tensor(ids, device="cuda")

    train, dev = load("train"), load("dev")
    steps = a.epochs * ((len(train) + a.bs - 1) // a.bs)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1, (s + 1) / 10) * max(0.0, 1 - s / steps))
    log, t0, step = [], time.time(), 0
    for ep in range(a.epochs):
        data = examples(train, rng)
        rng.shuffle(data)
        model.train()
        for i in range(0, len(data), a.bs):
            logits, y = last_logits(data[i:i + a.bs])
            loss = F.cross_entropy(logits, y)
            loss.backward(); opt.step(); sched.step(); opt.zero_grad()
            step += 1
            if step % 20 == 0:
                print(f"epoch {ep + 1} step {step}/{steps} loss {loss.item():.4f} ({time.time() - t0:.0f}s)", flush=True)
                log.append({"step": step, "loss": loss.item()})

    # Quick sanity on dev (original option order): argmax over ALL tokens must be the label.
    model.eval()
    ok = 0
    with torch.no_grad():
        ex = examples(dev, random.Random(1))
        for i in range(0, len(ex), a.bs):
            logits, y = last_logits(ex[i:i + a.bs])
            ok += (logits.argmax(-1) == y).sum().item()
    print(f"dev accuracy (shuffled options, argmax over full vocab): {ok / len(ex):.3f}")

    out.mkdir(parents=True, exist_ok=True)
    model.merge_and_unload().save_pretrained(out)
    tok.save_pretrained(out)
    (out / "train_log.json").write_text(json.dumps(
        {"base": a.base, "epochs": a.epochs, "bs": a.bs, "lr": a.lr, "n_train": len(train), "dev_acc": ok / len(ex),
         "seconds": time.time() - t0, "loss": log}, indent=1))
    print("saved", out)


if __name__ == "__main__":
    main()

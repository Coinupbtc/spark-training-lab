# Spark Training Lab

![Screenshot](docs/screenshots/hero.png)

![CI](https://github.com/Coinupbtc/spark-training-lab/actions/workflows/ci.yml/badge.svg)

LoRA / QLoRA **lab for NVIDIA DGX Spark**: adapters, datasets, merge scripts, and run logs.

Multi-GB base weights and merged GGUFs are **not** in this repo (rebuild locally).

## At a glance

| | |
|---|---|
| **What it is** | A **LoRA/QLoRA training lab** for DGX Spark: adapters, datasets, merge/eval scripts, and run logs. Multi-GB base weights are **not** in git. |
| **What it’s for** | Safe, small fine-tunes and adapter experiments without dumping huge checkpoints into GitHub — ship methods + adapters, rebuild merges locally. |
| **How to use it** | `./setup.sh` to orient; browse `adapters/` + `datasets/`. When you have a base model: `INSTALL_DEPS=1 ./setup.sh`, then follow `scripts/` and `docs/`. |

## Try it (pick one)

### One command
```bash
git clone https://github.com/Coinupbtc/spark-training-lab.git
cd spark-training-lab && ./setup.sh
```

### Copy-paste (browse only — no GPU)
```bash
git clone https://github.com/Coinupbtc/spark-training-lab.git && cd spark-training-lab
ls adapters datasets scripts runs
./setup.sh
```

### Train (when you have a base model)
```bash
INSTALL_DEPS=1 ./setup.sh
# put / point at your base weights, then follow scripts/ + docs/
ls scripts/
```

## What is / is not

| Do | Don't |
|----|-------|
| LoRA/QLoRA on 7B–35B class models | Expect full multi-GB weights in git |
| Small curated datasets (100–5k examples) | Dump unfiltered private corpora |
| Eval fixed prompts before/after | Ship an adapter with no smoke test |

## Layout

| Path | What |
|------|------|
| `adapters/` | Saved LoRA weights (~tens of MB) |
| `datasets/` | JSONL train/eval |
| `scripts/` | Prepare, train, merge, eval |
| `runs/` | Logs / metrics |
| `docs/` | Runbooks |
| `setup.sh` | One-command orientation |

## Related

- [miaai35-tune](https://github.com/Coinupbtc/miaai35-tune) — measured llama.cpp serving tune
- [zwell-bench](https://github.com/Coinupbtc/zwell-bench) — bakeoff harness
- [spark-console](https://github.com/Coinupbtc/spark-console) — local GPU/fleet UI


## License

MIT

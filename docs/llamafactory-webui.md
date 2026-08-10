# LLaMA-Factory WebUI (LLaMA Board) — Spark 2

## Open
- **LAN:** http://192.168.50.167:7860
- **On Spark2 desktop:** http://127.0.0.1:7860
- **SSH tunnel from laptop:**  
  `ssh -L 7860:127.0.0.1:7860 coinupbtc@192.168.50.167`  
  then open http://127.0.0.1:7860

## Service
```bash
ssh spark2
systemctl --user status llamafactory-webui
systemctl --user restart llamafactory-webui
systemctl --user stop llamafactory-webui
# or: ~/scripts/dgx/llamafactory-webui-start.sh
```

## Paths
- Code: `~/Documents/projects/LLaMA-Factory`
- Venv: `~/Documents/projects/factoryEnv`
- Sample dataset: `adam_sample` (registered in data/dataset_info.json)

## First train in the UI (click path)
1. **Language** → en
2. **Model name** → e.g. `Qwen/Qwen2.5-1.5B-Instruct` (small first) or `Qwen/Qwen2.5-7B-Instruct`
3. **Finetuning method** → `lora`
4. **Dataset** → `adam_sample` (or built-in `alpaca_en_demo` for a longer demo)
5. **Output dir** → `saves/adam_test`
6. **Train** tab → **Start**

Keep batch size small (1–2). Do not train while DSpark TP=2 is loaded.

## Install note
PyTorch 2.13 + CUDA 13 aarch64 wheels; LLaMA-Factory 0.9.6.dev0 (editable).

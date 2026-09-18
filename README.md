# alignment_eval

```
python -m venv .venv
source .venv/bin/activate
CMAKE_ARGS="-DGGML_METAL=on -DGGML_CCACHE=OFF" pip install -r requirements.txt
export OPENROUTER_API_KEY=sk-or-...
python eval.py input.json -o results.png --json output.json
python eval.py input_evil.json -o results_evil.png --json output_evil.json
```

`openrouter` models use its API. `huggingface` models run locally; set `base` for base checkpoints and `file` for GGUF repositories:

```json
{
  "models": [
    {"provider": "openrouter", "model": "openai/gpt-5.6-luna-pro"},
    {"provider": "huggingface", "model": "Qwen/Qwen3-0.6B-Base", "base": true},
    {"provider": "huggingface", "model": "org/model-GGUF", "file": "model.Q4_K_M.gguf"}
  ]
}
```
